"""End-to-end streaming transcription.

``run_streaming_transcription`` mirrors
``instrument_agnostic_amt.cli.infer_stem.run_stem_separated_transcription`` stage
for stage, but instead of returning one dict at the end it pushes JSON-ready
events into an ``emit`` callback:

    stage      a pipeline stage started / finished / was skipped
    progress   fraction of the current phase, for the progress bar and ETA
    note       one note (``final=False`` means it can still grow)
    stems      the stem names the separator produced
    beat_grid  tempo read back from the post-processed MIDI
    complete   finished MIDI, ready to download

The per-stem transcription streams because the windowed decoder accepts an
``on_window_consumed`` hook; see ``emitters.py`` for the finality rules.
"""

from __future__ import annotations

import math
import shutil
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from ..amt.cli import infer
from ..amt.inference.windowed import _build_window_starts
from ..beat_chord.cli.infer import predict_beat_chord_for_midi
from ..cli.infer_stem import (
    _move_model,
    get_beat_chord_models,
    get_refinement_models,
    get_stem_pipeline_models,
    get_velocity_models,
    merge_midis_logic,
    prepare_audio_for_stem_separation,
    refine_stem_instrument_midis,
    resolve_stem_model_type,
)
from ..runtime import is_amp_supported, resolve_amp_dtype, resolve_device
from ..taxonomy.instrument_classes import INSTRUMENT_CLASSES
from ..velocity.cli.infer_velocity import predict_velocity_for_stem_midis
from ..whisper_lyrics import add_whisper_lyrics_to_vocals_midi
from .emitters import make_emitter

EmitFn = Callable[[dict[str, Any]], None]
CancelFn = Callable[[], bool]


PHASE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("prepare", 0.01),
    ("separate", 0.34),
    ("transcribe", 0.45),
    ("post", 0.20),
)

# Announced up front so the UI can list every stage before any of them runs.
_EXPECTED_STEMS: tuple[str, ...] = (
    "bass",
    "drums",
    "guitar",
    "other",
    "piano",
    "vocals",
)
_POST_STAGES: tuple[tuple[str, str], ...] = (
    ("lyrics", "识别歌词"),
    ("refine", "乐器标签重标注"),
    ("merge", "合并 MIDI"),
    ("velocity", "预测力度"),
    ("beat_chord", "节拍 · 和弦 · 调性"),
    ("export", "导出文件"),
)

# Tempos outside this window are treated as artefacts of the beat decoder.
_MIN_BPM = 30.0
_MAX_BPM = 400.0


@dataclass
class StreamOptions:
    """Knobs for one streaming job."""

    window_batch_size: int = 1
    max_midi_melodic_instruments: int = 15
    merge_onset_ms: float = 50.0
    refine_instruments: bool = True
    refinement_stem_names: tuple[str, ...] = ("piano", "bass", "guitar")
    refinement_mode: str = "cluster"
    predict_velocity: bool = True
    predict_beat_chord: bool = True
    beat_chord_use_audio: bool = True
    transcribe_lyrics: bool = True
    lyrics_confidence_threshold: float = 0.65
    whisper_lyrics_language: str | None = None
    low_vram_mode: bool = True
    no_half: bool = False
    cleanup_separated_stems: bool = True
    amp: bool = False
    amp_dtype: str | None = None
    compile_model: bool = False
    compile_velocity: bool = False
    semi_crf_backend: str = "torch"
    transcribe_drum_stems: bool = True
    checkpoint_path: Path | None = None
    device: str | None = "auto"


@dataclass
class _JobState:
    options: StreamOptions
    job_dir: Path
    emit: EmitFn
    device: torch.device
    started_at: float = field(default_factory=time.perf_counter)
    stem_audio_paths: dict[str, Path] = field(default_factory=dict)
    stem_midi_paths: dict[str, Path] = field(default_factory=dict)
    note_count: int = 0

    def stage(
        self,
        key: str,
        label: str,
        status: str,
        *,
        detail: str | None = None,
        elapsed: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "stage",
            "key": key,
            "label": label,
            "status": status,
        }
        if detail:
            payload["detail"] = detail
        if elapsed is not None:
            payload["elapsed"] = round(float(elapsed), 3)
        if status == "running":
            payload["started_at"] = round(time.perf_counter() - self.started_at, 3)
        self.emit(payload)

    def progress(self, phase: str, fraction: float, **extra: Any) -> None:
        self.emit(
            {
                "type": "progress",
                "phase": phase,
                "fraction": max(0.0, min(1.0, float(fraction))),
                **extra,
            }
        )


def _instrument_label(instrument_id: int) -> str:
    try:
        return str(INSTRUMENT_CLASSES[int(instrument_id)])
    except (KeyError, IndexError, TypeError):
        return "unknown"


def _note_payload(event: Any) -> dict[str, Any]:
    return {
        "type": "note",
        "id": event.id,
        "instrument": event.instrument,
        "instrument_id": event.instrument_id,
        "pitch": event.pitch,
        "start": event.start,
        "end": event.end,
        "velocity": event.velocity,
        "final": event.final,
        "stem": event.stem,
    }


class _CountingSeparator(torch.nn.Module):
    """Wraps the separation model so chunk progress can be reported."""

    def __init__(self, inner: torch.nn.Module, on_batch: Callable[[int], None]) -> None:
        super().__init__()
        self.inner = inner
        self.on_batch = on_batch
        self.batches = 0

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        output = self.inner(batch)
        self.batches += 1
        self.on_batch(self.batches)
        return output


def _separation_batch_count(prepared_path: Path, sep_config: Any) -> int:
    """Reproduce ``stem_splitter``'s chunk arithmetic to size the progress bar."""
    info = sf.info(str(prepared_path))
    total_length = int(info.frames)
    if int(info.samplerate) != int(sep_config.target_sample_rate):
        total_length = int(
            round(total_length * int(sep_config.target_sample_rate) / int(info.samplerate))
        )
    chunk_size = int(sep_config.chunk_size)
    hop_size = int(sep_config.hop_size) if sep_config.hop_size else chunk_size // 2
    if total_length <= chunk_size:
        padded_length = chunk_size
    else:
        steps = math.ceil((total_length - chunk_size) / hop_size)
        padded_length = steps * hop_size + chunk_size
    starts = len(range(0, padded_length - chunk_size + 1, hop_size))
    return max(1, math.ceil(starts / max(1, int(sep_config.batch_size))))


def _read_tempo_map(midi_path: Path) -> dict[str, Any] | None:
    """Timeline of the finished MIDI: tempo, meters, beats, keys and chords.

    The beat model writes a "Predicted Tempo Map" track (tempo + time signature +
    key signature) and a "Predicted Chords" track (one marker per chord), so the
    roll can draw real bar lines and show the harmonic changes instead of
    assuming a constant metre. ``tempos`` is downsampled for the SSE payload;
    ``beats``/``downbeats`` are the grid the roll draws.
    """
    try:
        import mido

        midi = mido.MidiFile(str(midi_path))
        ppq = int(midi.ticks_per_beat)
        # NOTE: iterating a MidiFile yields *seconds*; merge_tracks keeps ticks,
        # which is what a beat grid has to be built from.
        tempo_points: list[tuple[int, int]] = [(0, 500_000)]
        meters: list[tuple[int, int, int]] = [(0, 4, 4)]
        keys: list[tuple[int, str]] = []
        chords: list[tuple[int, str]] = []
        tick = 0
        for message in mido.merge_tracks(midi.tracks):
            tick += int(message.time)
            if message.type == "set_tempo":
                tempo_points.append((tick, int(message.tempo)))
            elif message.type == "time_signature":
                signature = (int(message.numerator), int(message.denominator))
                if meters[-1][1:] != signature:
                    meters.append((tick, *signature))
            elif message.type == "key_signature":
                key = str(message.key)
                if not keys or keys[-1][1] != key:
                    keys.append((tick, key))
            elif message.type == "marker":
                chord = str(message.text).strip()
                if chord and (not chords or chords[-1][1] != chord):
                    chords.append((tick, chord))

        def seconds_for(target: int) -> float:
            total = 0.0
            previous_tick, previous_tempo = tempo_points[0]
            for point_tick, point_tempo in tempo_points:
                if point_tick >= target:
                    break
                total += mido.tick2second(
                    point_tick - previous_tick, ppq, previous_tempo
                )
                previous_tick, previous_tempo = point_tick, point_tempo
            return total + mido.tick2second(
                target - previous_tick, ppq, previous_tempo
            )

        # Real beat/bar grid: beats are aligned to each time-signature change, and
        # the bar accent follows that signature's numerator.
        beats: list[float] = []
        downbeats: list[float] = []
        total_ticks = tick
        for index, (start_tick, meter_num, meter_den) in enumerate(meters):
            end_tick = meters[index + 1][0] if index + 1 < len(meters) else total_ticks
            beat_ticks = ppq * 4 / max(1, meter_den)
            step = 0
            while True:
                position = start_tick + round(step * beat_ticks)
                if position >= end_tick:
                    break
                time = round(seconds_for(position), 4)
                beats.append(time)
                if step % max(1, meter_num) == 0:
                    downbeats.append(time)
                step += 1

        tempos: list[dict[str, float]] = []
        for point_tick, point_tempo in tempo_points:
            bpm = 60_000_000.0 / float(point_tempo)
            if tempos and abs(tempos[-1]["bpm"] - bpm) < 0.01:
                continue
            tempos.append(
                {"time": round(seconds_for(point_tick), 3), "bpm": round(bpm, 2)}
            )
        if not tempos:
            return None
        # The exported map can carry placeholder/mis-scaled tempos (e.g. 4-digit
        # BPM from a metric modulation artefact). Keep the grid and the reported
        # range inside a musical window; the raw values still ship for reference.
        musical = [entry for entry in tempos if _MIN_BPM <= entry["bpm"] <= _MAX_BPM]
        if not musical:
            musical = tempos
        values = sorted(entry["bpm"] for entry in musical)

        def percentile(fraction: float) -> float:
            index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
            return float(values[index])

        median = percentile(0.5)
        stride = max(1, len(musical) // 240)
        return {
            "bpm": round(median, 2),
            "bpm_min": round(percentile(0.05), 2),
            "bpm_max": round(percentile(0.95), 2),
            "beats_per_bar": meters[0][1] or 4,
            "tempos": musical[::stride],
            "beats": beats[:6000],
            "downbeats": downbeats[:2000],
            "meters": [
                {
                    "time": round(seconds_for(point_tick), 3),
                    "numerator": meter_num,
                    "denominator": meter_den,
                }
                for point_tick, meter_num, meter_den in meters
            ],
            "keys": [{"time": round(seconds_for(t), 3), "key": k} for t, k in keys],
            "chords": [
                {"time": round(seconds_for(t), 3), "chord": c}
                for t, c in chords[:1200]
            ],
        }
    except Exception:  # pragma: no cover - tempo map is a nicety, never fatal
        return None


def run_streaming_transcription(
    audio_path: Path | str,
    emit: EmitFn,
    *,
    job_dir: Path | str,
    options: StreamOptions | None = None,
    should_cancel: CancelFn | None = None,
    model_provider: Callable[[str, bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline, streaming events through ``emit``.

    ``model_provider`` lets the server inject already-warm models; by default the
    models are loaded lazily through ``get_stem_pipeline_models``.
    """
    options = options or StreamOptions()
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    job_root = Path(job_dir)
    job_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(options.device)
    state = _JobState(
        options=options, job_dir=job_root, emit=emit, device=device
    )
    cancelled = should_cancel or (lambda: False)

    def check_cancel() -> None:
        if cancelled():
            raise _Cancelled()

    emit(
        {
            "type": "hello",
            "audio": audio_file.name,
            "phases": [{"key": key, "weight": weight} for key, weight in PHASE_WEIGHTS],
            "device": str(device),
            "stems": list(_EXPECTED_STEMS),
            "post_stages": list(_POST_STAGES),
        }
    )
    _run_pipeline(state, audio_file, options, model_provider, check_cancel)
    return {
        "job_dir": str(job_root),
        "note_count": int(state.note_count),
        "stem_count": len(state.stem_audio_paths),
    }


class _Cancelled(Exception):
    """Raised internally when the client goes away mid-job."""


def _run_pipeline(
    state: _JobState,
    audio_file: Path,
    options: StreamOptions,
    model_provider: Callable[[str, bool], dict[str, Any]] | None,
    check_cancel: Callable[[], None],
) -> None:
    emit = state.emit
    device = state.device
    job_root = state.job_dir
    stem_dir = job_root / "stems"
    stem_midi_dir = job_root / "stem_midis"
    refined_stem_midi_dir = job_root / "refined_stem_midis"
    merged_dir = job_root / "merged"
    for directory in (stem_dir, stem_midi_dir, merged_dir):
        directory.mkdir(parents=True, exist_ok=True)

    provider = model_provider or (
        lambda model_type, low_vram: get_stem_pipeline_models(
            checkpoint_path=options.checkpoint_path,
            device_preference=options.device,
            model_type=model_type,
            low_vram_mode=low_vram,
            no_half=options.no_half,
            compile_model=options.compile_model,
            semi_crf_backend=options.semi_crf_backend,
        )
    )

    # ── 1. prepare ───────────────────────────────────────────────────────────
    state.stage("prepare", "解码音频", "running")
    started = time.perf_counter()
    separation_input = prepare_audio_for_stem_separation(
        audio_file, temp_dir=job_root / "prepared_inputs"
    )
    state.stage("prepare", "解码音频", "done", elapsed=time.perf_counter() - started)
    state.progress("prepare", 1.0)

    # ── 2. stem separation ───────────────────────────────────────────────────
    bundle = provider("default", bool(options.low_vram_mode))
    sep_config = bundle["sep_config"]
    sep_model = bundle["sep_model"]
    sep_dtype = bundle["sep_dtype"]
    total_batches = _separation_batch_count(separation_input, sep_config)
    state.stage(
        "separate",
        "分离音轨",
        "running",
        detail=f"{len(sep_config.stem_names)} 轨 · {total_batches} 个分块",
    )
    started = time.perf_counter()

    def on_batch(completed: int) -> None:
        state.progress("separate", completed / max(1, total_batches), completed=completed, total=total_batches)

    if options.low_vram_mode:
        _move_model(sep_model, device)
    try:
        from stem_splitter.inference import _separate_one_file

        stems = _separate_one_file(
            separation_input,
            stem_dir,
            sep_config,
            _CountingSeparator(sep_model, on_batch),
            device,
            sep_dtype,
        )
    finally:
        if options.low_vram_mode:
            _move_model(sep_model, "cpu")
    state.stage(
        "separate",
        "分离音轨",
        "done",
        elapsed=time.perf_counter() - started,
        detail=f"{len(stems)} 轨",
    )
    state.progress("separate", 1.0)
    check_cancel()
    if not stems:
        raise RuntimeError("Stem separation produced no stems")
    state.stem_audio_paths = {name: Path(path) for name, path in stems.items()}
    emit({"type": "stems", "stems": sorted(state.stem_audio_paths)})

    # ── 3. per-stem transcription ────────────────────────────────────────────
    amt_amp_enabled = bool(options.amp and is_amp_supported(device))
    amt_amp_dtype = resolve_amp_dtype(device, options.amp_dtype)
    stem_names = sorted(state.stem_audio_paths)
    waveform_cache: dict[Path, torch.Tensor] = {}

    def load_stem_waveform(path: Path, sample_rate: int) -> torch.Tensor:
        resolved = Path(path).resolve()
        cached = waveform_cache.get(resolved)
        if cached is None or int(cached.shape[-1]) == 0:
            cached, _, _ = infer._load_audio(resolved, target_sample_rate=int(sample_rate))
            waveform_cache[resolved] = cached
        return cached

    for stem_index, stem_name in enumerate(stem_names):
        check_cancel()
        stem_path = state.stem_audio_paths[stem_name]
        output_midi = stem_midi_dir / f"{audio_file.stem}_{stem_name}.mid"
        if "drum" in stem_name.lower() and not options.transcribe_drum_stems:
            continue

        model_type = resolve_stem_model_type(stem_name)
        stage_key = f"transcribe:{stem_name}"
        state.stage(stage_key, f"识别 {stem_name}", "running", detail=f"model={model_type}")
        started = time.perf_counter()
        current_bundle = provider(model_type, bool(options.low_vram_mode))
        amt_model = current_bundle["amt_model"]
        amt_forward = current_bundle["amt_forward"]
        amt_config = current_bundle["amt_config"]
        amt_settings = current_bundle["amt_settings"]
        allowed_ids = infer.filter_supported_instrument_class_ids(
            infer.resolve_stem_instrument_class_ids(stem_name),
            num_model_classes=amt_config.num_instrument_classes,
        )
        waveform = load_stem_waveform(stem_path, int(amt_config.sample_rate))
        sample_rate = int(amt_config.sample_rate)
        stem_emit_limit = len(
            _build_window_starts(
                total_audio_frames=int(waveform.shape[-1]),
                window_audio_frames=int(
                    round(amt_settings.window_ms * sample_rate / 1000.0)
                ),
                stride_audio_frames=int(
                    round(amt_settings.stride_ms * sample_rate / 1000.0)
                ),
            )
        )
        merge_gap_samples = (
            int(amt_config.hop_length)
            if amt_settings.merge_gap_ms is None
            else int(round(float(amt_settings.merge_gap_ms) * sample_rate / 1000.0))
        )
        merge_onset_samples = int(
            round(float(amt_settings.merge_onset_ms) * sample_rate / 1000.0)
        )
        emitter_holder: dict[str, Any] = {"emitter": None, "windows": 0}

        def on_window(stitcher: Any, window_start: int, *, _stem=stem_name) -> None:
            emitter = emitter_holder["emitter"]
            if emitter is None:
                emitter = make_emitter(
                    amt_config.semi_crf_version,
                    stitcher,
                    stem=_stem,
                    sample_rate=sample_rate,
                    merge_gap_samples=merge_gap_samples,
                    merge_onset_samples=merge_onset_samples,
                    instrument_label=_instrument_label,
                )
                emitter_holder["emitter"] = emitter
            for note_event in emitter.on_window(int(window_start)):
                state.note_count += 1
                emit(_note_payload(note_event))
            emitter_holder["windows"] = emitter_holder.get("windows", 0) + 1
            completed = int(emitter_holder["windows"])
            fraction = (stem_index + min(1.0, completed / max(1, stem_emit_limit))) / max(
                1, len(stem_names)
            )
            state.progress(
                "transcribe",
                fraction,
                stem=_stem,
                completed=completed,
                total=stem_emit_limit,
            )

        if options.low_vram_mode:
            _move_model(amt_model, device)
        try:
            notes, _, _ = infer.run_inference(
                model=amt_model,
                forward_model=amt_forward,
                waveform=waveform.to(device),
                model_config=amt_config,
                settings=amt_settings,
                device=device,
                amp_enabled=amt_amp_enabled,
                amp_dtype=amt_amp_dtype,
                velocity=100,
                merge_gap_ms=None,
                merge_onset_ms=options.merge_onset_ms,
                silence_gate_rms_dbfs=-72,
                window_batch_size=int(options.window_batch_size),
                max_midi_melodic_instruments=options.max_midi_melodic_instruments,
                disable_tqdm=True,
                allowed_instrument_ids=allowed_ids,
                on_window_consumed=on_window,
            )
            emitter = emitter_holder["emitter"]
            if emitter is not None:
                for note_event in emitter.finish():
                    state.note_count += 1
                    emit(_note_payload(note_event))
            instrument_volumes = (
                None
                if options.predict_velocity
                else dict(infer.DEFAULT_INSTRUMENT_VOLUMES)
            )
            midi = infer._build_midi(
                notes,
                sample_rate=sample_rate,
                instrument_volumes=instrument_volumes,
            )
            midi.write(str(output_midi))
            state.stem_midi_paths[stem_name] = output_midi
        finally:
            if options.low_vram_mode:
                _move_model(amt_model, "cpu")
        state.stage(
            stage_key,
            f"识别 {stem_name}",
            "done",
            elapsed=time.perf_counter() - started,
            detail=f"{state.note_count} 个音符（累计）",
        )
        state.progress(
            "transcribe", (stem_index + 1) / max(1, len(stem_names)), stem=stem_name
        )

        if "vocal" in stem_name.lower() and options.transcribe_lyrics:
            state.stage("lyrics", "识别歌词", "running")
            started = time.perf_counter()
            try:
                add_whisper_lyrics_to_vocals_midi(
                    vocals_wav_path=stem_path,
                    stem_midi_path=output_midi,
                    language=options.whisper_lyrics_language,
                    target_track_name="vocals",
                    lyrics_confidence_threshold=options.lyrics_confidence_threshold,
                )
                state.stage(
                    "lyrics", "识别歌词", "done", elapsed=time.perf_counter() - started
                )
            except Exception as err:  # pragma: no cover - optional enrichment
                state.stage(
                    "lyrics", "识别歌词", "skipped", detail=str(err)[:200]
                )
        elif "vocal" in stem_name.lower():
            state.stage("lyrics", "识别歌词", "skipped", detail="已关闭")

    check_cancel()
    if not state.stem_midi_paths:
        raise RuntimeError("No stem MIDI files were generated")
    waveform_cache.clear()

    # ── 4. instrument refinement ─────────────────────────────────────────────
    if options.refine_instruments:
        state.stage("refine", "乐器标签重标注", "running")
        started = time.perf_counter()
        try:
            refinement_bundle = get_refinement_models(
                checkpoint_path=None,
                device_preference=device,
                low_vram_mode=bool(options.low_vram_mode),
            )
            refined = refine_stem_instrument_midis(
                stem_midis=state.stem_midi_paths,
                stem_audios=state.stem_audio_paths,
                output_dir=refined_stem_midi_dir,
                refinement_model=refinement_bundle["refinement_model"],
                refinement_config=refinement_bundle["refinement_config"],
                device=device,
                low_vram_mode=bool(options.low_vram_mode),
                stem_names=list(options.refinement_stem_names),
                mode=options.refinement_mode,
                window_batch_size=int(options.window_batch_size),
                disable_tqdm=True,
                waveform_loader=load_stem_waveform,
                amp=amt_amp_enabled,
                amp_dtype=amt_amp_dtype,
            )
            if refined:
                state.stem_midi_paths.update({k: Path(v) for k, v in refined.items()})
                state.stage(
                    "refine",
                    "乐器标签重标注",
                    "done",
                    elapsed=time.perf_counter() - started,
                    detail=f"{len(refined)} 轨已修正",
                )
            else:
                state.stage(
                    "refine", "乐器标签重标注", "skipped", detail="没有可修正的音轨"
                )
        except Exception as err:
            state.stage("refine", "乐器标签重标注", "error", detail=str(err)[:200])
    else:
        state.stage("refine", "乐器标签重标注", "skipped", detail="已关闭")

    # ── 5. merge ─────────────────────────────────────────────────────────────
    state.stage("merge", "合并 MIDI", "running")
    started = time.perf_counter()
    merged_midi_path = merged_dir / f"{audio_file.stem}.mid"
    merge_midis_logic(
        [state.stem_midi_paths[name] for name in sorted(state.stem_midi_paths)],
        merged_midi_path,
        max_melodic_instruments=options.max_midi_melodic_instruments,
    )
    state.stage("merge", "合并 MIDI", "done", elapsed=time.perf_counter() - started)

    # ── 6. velocity ──────────────────────────────────────────────────────────
    if options.predict_velocity:
        state.stage("velocity", "预测力度", "running")
        started = time.perf_counter()
        velocity_model = None
        try:
            velocity_model, velocity_forward, velocity_config = get_velocity_models(
                checkpoint_path=None,
                device_preference="cpu" if options.low_vram_mode else device,
                compile_velocity=options.compile_velocity,
                compile_mode="default",
            )
            if options.low_vram_mode:
                _move_model(velocity_model, device)
            preloaded_waveforms = {
                name: load_stem_waveform(path, int(velocity_config.sample_rate))
                for name, path in state.stem_audio_paths.items()
                if Path(path).exists()
            }
            velocity_midi_path = merged_dir / f"{audio_file.stem}_velocity.mid"
            predict_velocity_for_stem_midis(
                stem_midis={
                    name: path
                    for name, path in state.stem_midi_paths.items()
                    if Path(path).exists()
                },
                stem_audios=state.stem_audio_paths,
                output_midi_path=velocity_midi_path,
                template_midi_path=merged_midi_path,
                device=device,
                window_seconds=8.0,
                max_melodic_instruments=options.max_midi_melodic_instruments,
                disable_tqdm=True,
                compile_velocity=options.compile_velocity,
                compile_mode="default",
                preloaded_model=velocity_model,
                preloaded_config=velocity_config,
                preloaded_forward=velocity_forward,
                preloaded_waveforms=preloaded_waveforms,
                amp=amt_amp_enabled,
                amp_dtype=amt_amp_dtype,
            )
            merged_midi_path = velocity_midi_path
            state.stage(
                "velocity", "预测力度", "done", elapsed=time.perf_counter() - started
            )
        except Exception as err:
            state.stage("velocity", "预测力度", "error", detail=str(err)[:200])
        finally:
            if options.low_vram_mode and velocity_model is not None:
                _move_model(velocity_model, "cpu")
    else:
        state.stage("velocity", "预测力度", "skipped", detail="已关闭")

    # ── 7. beat / chord / key ────────────────────────────────────────────────
    if options.predict_beat_chord:
        state.stage("beat_chord", "节拍 · 和弦 · 调性", "running")
        started = time.perf_counter()
        try:
            (
                beat_chord_model,
                beat_chord_config,
                beat_chord_metadata,
            ) = get_beat_chord_models(
                checkpoint_path=None,
                device_preference=device,
            )
            beat_chord_midi_path = merged_dir / f"{audio_file.stem}_beat_chord.mid"
            predict_beat_chord_for_midi(
                input_midi_path=merged_midi_path,
                output_midi_path=beat_chord_midi_path,
                device=device,
                window_batch_size=int(options.window_batch_size),
                preloaded_model=beat_chord_model,
                preloaded_model_config=beat_chord_config,
                preloaded_metadata=beat_chord_metadata,
                audio_path=audio_file if options.beat_chord_use_audio else None,
            )
            merged_midi_path = beat_chord_midi_path
            tempo_map = _read_tempo_map(merged_midi_path)
            if tempo_map:
                emit({"type": "beat_grid", **tempo_map})
            state.stage(
                "beat_chord",
                "节拍 · 和弦 · 调性",
                "done",
                elapsed=time.perf_counter() - started,
                detail=(
                    f"{tempo_map['bpm']:.0f} BPM · {tempo_map['beats_per_bar']}/4"
                    if tempo_map
                    else None
                ),
            )
        except Exception as err:
            state.stage("beat_chord", "节拍 · 和弦 · 调性", "error", detail=str(err)[:200])
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        state.stage("beat_chord", "节拍 · 和弦 · 调性", "skipped", detail="已关闭")

    # ── 8. export ────────────────────────────────────────────────────────────
    state.stage("export", "导出文件", "running")
    export_dir = job_root / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    exported = export_dir / f"{audio_file.stem}.mid"
    shutil.copyfile(merged_midi_path, exported)
    state.stage("export", "导出文件", "done", detail=exported.name)
    if options.cleanup_separated_stems:
        shutil.rmtree(stem_dir, ignore_errors=True)
    state.progress("post", 1.0)
    emit(
        {
            "type": "complete",
            "midi": exported.name,
            "note_count": int(state.note_count),
            "stem_count": len(state.stem_audio_paths),
            "stems": sorted(state.stem_audio_paths),
            "elapsed": round(time.perf_counter() - state.started_at, 3),
        }
    )


def format_error(exc: BaseException) -> dict[str, Any]:
    """Standard error event body."""
    return {
        "type": "error",
        "message": str(exc)[:500] or exc.__class__.__name__,
        "traceback": traceback.format_exc()[-2000:],
    }
