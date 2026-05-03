from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable

import os
import pretty_midi
import soundfile as sf
import torch
import torch.hub
import torchaudio.functional as audio_functional
from tqdm.auto import tqdm

from models.model import AudioSemiCRFTransformer, SemiCRFModelConfig
from models.semi_crf import decode_pitch_intervals
from dataset import MIN_MIDI_PITCH


DEFAULT_CHECKPOINT_URL = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model.pth?download=true"
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/best_model.pth")


@dataclass
class PredictedNote:
    pitch: int
    start_frame: int
    end_frame: int
    velocity: int
    has_onset: bool = True
    has_offset: bool = True
    instrument_id: int = 0
    instrument_candidates: tuple[int, ...] = ()


@dataclass(frozen=True)
class InferenceSettings:
    window_ms: int
    stride_ms: int
    track_batch_size: int
    length_scaling: str
    length_penalty: float


def resolve_amp_dtype(device: torch.device, dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Semi-CRF AMT inference on an audio file and export MIDI.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Path to the trained model checkpoint (.pth). If not provided, downloads from Hugging Face to {DEFAULT_CHECKPOINT_PATH}.",
    )
    parser.add_argument(
        "--audio", type=Path, required=True, help="Path to the input audio file"
    )
    parser.add_argument(
        "--output-midi",
        type=Path,
        default=None,
        help="Output MIDI path. Defaults to <audio_stem>.mid next to the input audio.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable Mixed Precision inference"
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="bf16" if torch.cuda.is_bf16_supported() else "fp16",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=None,
        help="Inference window size in milliseconds. Defaults to the training window.",
    )
    parser.add_argument(
        "--stride-ms",
        type=int,
        default=None,
        help="Inference stride in milliseconds. Defaults to half of window-ms.",
    )
    parser.add_argument(
        "--semi-crf-track-batch-size",
        type=int,
        default=None,
        help="Chunk size for CRF decoding. Defaults to the training setting.",
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=1,
        help="Number of windows to process at once. Increase only if VRAM allows it.",
    )
    parser.add_argument(
        "--merge-gap-ms",
        type=float,
        default=None,
        help="Merge same-pitch notes when the gap is below this threshold. Defaults to one hop.",
    )
    parser.add_argument(
        "--merge-onset-ms",
        type=float,
        default=20.0,
        help="Merge same-pitch notes when their onset difference is below this threshold.",
    )
    parser.add_argument(
        "--velocity",
        type=int,
        default=100,
        help="Constant MIDI velocity for predicted notes.",
    )
    parser.add_argument(
        "--max-midi-melodic-instruments",
        type=int,
        default=15,
        help=(
            "Maximum number of non-drum instrument tracks in exported MIDI. "
            "Extra low-note-count instruments are reassigned by instrument_logits rank. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--silence-gate-rms-dbfs",
        type=float,
        default=-72,
        help="Skip fully silent windows before model forward pass.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser.parse_args()

def _ensure_checkpoint(checkpoint_path: Path | None) -> Path:
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}. Downloading from Hugging Face...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(DEFAULT_CHECKPOINT_URL, str(checkpoint_path))

    return checkpoint_path



def _iter_batches(values: list[int], batch_size: int) -> Iterable[list[int]]:
    chunk_size = max(1, int(batch_size))
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _coerce_model_config(raw_model_config: dict[str, Any]) -> SemiCRFModelConfig:
    allowed_fields = {field.name for field in fields(SemiCRFModelConfig)}
    kwargs = {
        key: value for key, value in raw_model_config.items() if key in allowed_fields
    }
    # 推論時に augmentation や gradient checkpoint は不要なので無効化する。
    kwargs["spec_augment_params"] = None
    kwargs["use_gradient_checkpoint"] = False
    return SemiCRFModelConfig(**kwargs)


def _load_model_and_settings(
    checkpoint_path: Path,
    *,
    device: torch.device,
    window_ms_override: int | None,
    stride_ms_override: int | None,
    track_batch_size_override: int | None,
) -> tuple[AudioSemiCRFTransformer, SemiCRFModelConfig, InferenceSettings]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    raw_model_config = None
    raw_run_config = checkpoint.get("config")
    if isinstance(raw_run_config, dict):
        raw_model_config = raw_run_config.get("model_config")
    if raw_model_config is None:
        raw_model_config = checkpoint.get("model_config")

    if not isinstance(raw_model_config, dict):
        import logging

        logging.warning(
            "Checkpoint does not contain model_config. Using fallback default config (sample_rate=22050, hop_length=512)."
        )
        raw_model_config = {
            "sample_rate": 22050,
            "hop_length": 512,
            "n_fft": 2048,
            "cqt_n_bins": 312,
        }

    model_config = _coerce_model_config(raw_model_config)
    model = AudioSemiCRFTransformer(model_config)

    raw_state_dict = checkpoint.get("model_state_dict")
    if raw_state_dict is not None:
        state_dict = raw_state_dict
    elif all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state_dict = checkpoint
    else:
        state_dict = None
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a state_dict: {checkpoint_path}")

    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    raw_args = (
        raw_run_config.get("args", {}) if isinstance(raw_run_config, dict) else {}
    )
    default_window_ms = int(raw_args.get("window_ms", 5000))
    window_ms = (
        int(window_ms_override) if window_ms_override is not None else default_window_ms
    )
    stride_ms = (
        int(stride_ms_override)
        if stride_ms_override is not None
        else max(1, window_ms // 2)
    )
    track_batch_size = (
        int(track_batch_size_override) if track_batch_size_override is not None else 128
    )
    settings = InferenceSettings(
        window_ms=window_ms,
        stride_ms=stride_ms,
        track_batch_size=track_batch_size,
        length_scaling=str(model_config.semi_crf_length_scaling),
        length_penalty=float(model_config.semi_crf_length_penalty),
    )
    return model, model_config, settings


def _load_audio(
    audio_path: Path,
    *,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int, int]:
    waveform_np, source_sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(waveform_np.T.copy())
    source_channels = int(waveform.shape[0])

    if source_channels <= 0:
        raise ValueError(f"Audio file has no channels: {audio_path}")
    if source_channels == 1:
        waveform = waveform.repeat(2, 1)
    elif source_channels > 2:
        waveform = waveform[:2]

    if int(source_sample_rate) != int(target_sample_rate):
        waveform = audio_functional.resample(
            waveform,
            int(source_sample_rate),
            int(target_sample_rate),
        )
    return waveform.contiguous(), int(source_sample_rate), source_channels


def _build_window_starts(
    *,
    total_audio_frames: int,
    window_audio_frames: int,
    stride_audio_frames: int,
) -> list[int]:
    if total_audio_frames <= window_audio_frames:
        return [0]
    starts = list(
        range(
            0,
            max(1, total_audio_frames - window_audio_frames + 1),
            stride_audio_frames,
        )
    )
    last_start = total_audio_frames - window_audio_frames
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _slice_window(
    waveform: torch.Tensor,
    *,
    start_frame: int,
    window_audio_frames: int,
) -> tuple[torch.Tensor, int]:
    end_frame = min(int(waveform.shape[-1]), start_frame + window_audio_frames)
    window = waveform[:, start_frame:end_frame]
    valid_audio_frames = int(window.shape[-1])
    if valid_audio_frames < window_audio_frames:
        padded = torch.zeros(
            (int(waveform.shape[0]), window_audio_frames),
            dtype=waveform.dtype,
        )
        padded[:, :valid_audio_frames] = window
        window = padded
    return window.contiguous(), valid_audio_frames


def _silence_gate_rms_linear(silence_gate_rms_dbfs: float | None) -> float | None:
    if silence_gate_rms_dbfs is None:
        return None
    threshold_dbfs = float(silence_gate_rms_dbfs)
    if threshold_dbfs > 0.0:
        raise ValueError("silence_gate_rms_dbfs must be <= 0 dBFS")
    return float(10.0 ** (threshold_dbfs / 20.0))


def _compute_silent_window_mask(
    batch_waveform: torch.Tensor,
    *,
    silence_gate_rms_linear: float | None,
) -> torch.Tensor | None:
    if silence_gate_rms_linear is None:
        return None
    if batch_waveform.dim() != 3:
        raise ValueError("batch_waveform must have shape [B, C, T]")
    # convert to mono for RMS calculation
    gate_input = batch_waveform.mean(dim=1, keepdim=True)
    rms = gate_input.float().square().mean(dim=-1).sqrt()
    return torch.all(rms < float(silence_gate_rms_linear), dim=1)


def _intervals_to_notes(
    intervals_by_pitch: list[list[tuple[int, int]]],
    *,
    boundary_flags_by_pitch: list[list[tuple[bool, bool, float, float]]] | None,
    instrument_logits: torch.Tensor | None,
    window_start_frame: int,
    valid_audio_frames: int,
    total_audio_frames: int,
    hop_length: int,
    velocity: int,
) -> list[PredictedNote]:
    notes: list[PredictedNote] = []
    window_valid_end = min(total_audio_frames, window_start_frame + valid_audio_frames)
    for pitch_index, pitch_intervals in enumerate(intervals_by_pitch):
        midi_pitch = MIN_MIDI_PITCH + pitch_index
        pitch_boundary_flags = (
            boundary_flags_by_pitch[pitch_index]
            if boundary_flags_by_pitch is not None
            else []
        )
        for interval_index, (begin_frame, end_frame) in enumerate(pitch_intervals):
            has_onset, has_offset, onset_off, offset_off = (
                pitch_boundary_flags[interval_index]
                if interval_index < len(pitch_boundary_flags)
                else (True, True, 0.0, 1.0)  # offset default fallback
            )

            # Use real float offset
            start_frame = window_start_frame + int(
                round((float(begin_frame) + float(onset_off)) * float(hop_length))
            )
            end_frame_exclusive = window_start_frame + min(
                valid_audio_frames,
                int(round((float(end_frame) + float(offset_off)) * float(hop_length))),
            )

            start_frame = max(0, min(start_frame, total_audio_frames))
            end_frame_exclusive = max(
                start_frame + 1,
                min(end_frame_exclusive, total_audio_frames),
            )
            if start_frame >= window_valid_end:
                continue

            if instrument_logits is not None:
                interval_start = int(begin_frame)
                interval_end = min(int(end_frame), valid_audio_frames)
                if interval_end > interval_start:
                    note_logits = instrument_logits[
                        interval_start:interval_end, pitch_index, :
                    ]
                    note_probs = torch.sigmoid(note_logits).mean(dim=0)
                    instrument_candidates = tuple(
                        int(idx)
                        for idx in torch.argsort(note_probs, descending=True).tolist()
                    )
                    instrument_id = instrument_candidates[0]
                else:
                    instrument_id = 0
                    instrument_candidates = (0,)
            else:
                instrument_id = 0
                instrument_candidates = (0,)

            notes.append(
                PredictedNote(
                    pitch=int(midi_pitch),
                    start_frame=int(start_frame),
                    end_frame=int(end_frame_exclusive),
                    velocity=int(velocity),
                    has_onset=bool(has_onset),
                    has_offset=bool(has_offset),
                    instrument_id=instrument_id,
                    instrument_candidates=instrument_candidates,
                )
            )
    return sorted(
        notes, key=lambda note: (note.start_frame, note.end_frame, note.pitch)
    )


def _decode_boundary_features(
    boundary_logits: torch.Tensor,
    entries: list[tuple[int, int, int, int, int]],
    *,
    batch_size: int,
    num_pitches: int,
) -> list[list[list[tuple[bool, bool, float, float]]]]:
    flags: list[list[list[tuple[bool, bool, float, float]]]] = [
        [[] for _ in range(num_pitches)] for _ in range(batch_size)
    ]
    if not entries:
        return flags

    presence_logits, offset_logits = boundary_logits.chunk(2, dim=-1)
    boundary_presence = presence_logits > 0.0

    offset_dist = torch.distributions.ContinuousBernoulli(logits=offset_logits)
    offset_values = (offset_dist.mean - 0.005) / 0.99
    offset_values = torch.clamp(offset_values, 0.0, 1.0)

    for row_index, entry in enumerate(entries):
        batch_index, pitch_index, _, _, _ = entry
        flags[batch_index][pitch_index].append(
            (
                bool(boundary_presence[row_index, 0].item()),
                bool(boundary_presence[row_index, 1].item()),
                float(offset_values[row_index, 0].item()),
                float(offset_values[row_index, 1].item()),
            )
        )
    return flags


def _merge_notes(
    notes: list[PredictedNote],
    *,
    merge_gap_frames: int,
    merge_onset_frames: int,
) -> list[PredictedNote]:
    if not notes:
        return []

    ordered_notes = sorted(
        notes,
        key=lambda note: (
            note.pitch,
            note.instrument_id,
            note.start_frame,
            note.end_frame,
        ),
    )
    merged: list[PredictedNote] = []
    current = replace(ordered_notes[0])
    for note in ordered_notes[1:]:
        can_merge_by_gap = (
            note.pitch == current.pitch
            and note.instrument_id == current.instrument_id
            and note.start_frame <= current.end_frame + int(merge_gap_frames)
            and not note.has_onset
            and not current.has_offset
        )
        can_merge_by_onset = (
            note.pitch == current.pitch
            and note.instrument_id == current.instrument_id
            and abs(note.start_frame - current.start_frame) <= int(merge_onset_frames)
        )

        if can_merge_by_gap or can_merge_by_onset:
            current.end_frame = max(current.end_frame, note.end_frame)
            current.velocity = max(current.velocity, note.velocity)
            current.has_onset = current.has_onset or note.has_onset
            current.has_offset = current.has_offset or note.has_offset
            continue
        merged.append(current)
        current = replace(note)
    merged.append(current)
    return sorted(
        merged,
        key=lambda note: (
            note.start_frame,
            note.pitch,
            note.instrument_id,
            note.end_frame,
        ),
    )


def _stitch_boundary_aware_notes(
    notes: list[PredictedNote],
) -> list[PredictedNote]:
    if not notes:
        return []

    return sorted(
        notes,
        key=lambda note: (
            note.start_frame,
            note.end_frame,
            note.pitch,
            note.instrument_id,
        ),
    )


def _truncate_overlapping_notes(
    notes: list[PredictedNote],
    *,
    separate_adjacent_same_pitch: bool = False,
    min_separation_frames: int = 0,
) -> list[PredictedNote]:
    if not notes:
        return []

    ordered_notes = sorted(
        notes,
        key=lambda note: (
            note.pitch,
            note.instrument_id,
            note.start_frame,
            note.end_frame,
        ),
    )

    by_pitch: dict[tuple[int, int], list[PredictedNote]] = {}
    for note in ordered_notes:
        key = (int(note.pitch), int(note.instrument_id))
        pitch_notes = by_pitch.setdefault(key, [])
        if pitch_notes:
            previous_note = pitch_notes[-1]
            separation_frames = max(
                1 if separate_adjacent_same_pitch else 0,
                int(min_separation_frames),
            )
            has_conflicting_boundary = (
                previous_note.end_frame > note.start_frame - separation_frames
                if separate_adjacent_same_pitch
                else previous_note.end_frame > note.start_frame
            )
            if has_conflicting_boundary:
                # Keep same-pitch note-off safely before the next note-on in the
                # exported MIDI track. A one-sample gap can quantize back to the
                # same MIDI tick, which some consumers read as a sustaining note.
                new_end_frame = int(note.start_frame) - separation_frames
                if new_end_frame > previous_note.start_frame:
                    pitch_notes[-1] = replace(
                        previous_note,
                        end_frame=new_end_frame,
                        has_offset=True,
                    )
                else:
                    pitch_notes.pop()
        pitch_notes.append(note)

    valid_notes = [
        note
        for pitch_notes in by_pitch.values()
        for note in pitch_notes
        if note.start_frame < note.end_frame
    ]
    return sorted(
        valid_notes,
        key=lambda note: (
            note.start_frame,
            note.pitch,
            note.instrument_id,
            note.end_frame,
        ),
    )


def _remap_overflow_midi_instruments(
    notes: list[PredictedNote],
    *,
    max_melodic_instruments: int,
) -> tuple[list[PredictedNote], dict[str, int]]:
    if not notes or max_melodic_instruments <= 0:
        return notes, {
            "midi_instrument_count_before_remap": len(
                {int(note.instrument_id) for note in notes}
            ),
            "midi_instrument_count_after_remap": len(
                {int(note.instrument_id) for note in notes}
            ),
            "remapped_instrument_count": 0,
            "remapped_note_count": 0,
        }

    from instrument_classes import INSTRUMENT_CLASSES

    drum_ids = {
        idx
        for idx, class_name in enumerate(INSTRUMENT_CLASSES)
        if class_name.lower() == "drums"
    }
    counts = Counter(int(note.instrument_id) for note in notes)
    melodic_ids = [inst_id for inst_id in counts if inst_id not in drum_ids]
    if len(melodic_ids) <= max_melodic_instruments:
        return notes, {
            "midi_instrument_count_before_remap": len(counts),
            "midi_instrument_count_after_remap": len(counts),
            "remapped_instrument_count": 0,
            "remapped_note_count": 0,
        }

    kept_melodic_ids = set(
        sorted(melodic_ids, key=lambda inst_id: (-counts[inst_id], inst_id))[
            :max_melodic_instruments
        ]
    )
    overflow_ids = set(melodic_ids) - kept_melodic_ids
    fallback_inst_id = max(kept_melodic_ids, key=lambda inst_id: counts[inst_id])

    remapped_notes: list[PredictedNote] = []
    remapped_note_count = 0
    for note in notes:
        inst_id = int(note.instrument_id)
        if inst_id not in overflow_ids:
            remapped_notes.append(note)
            continue

        replacement_inst_id = None
        for candidate_id in note.instrument_candidates:
            candidate_id = int(candidate_id)
            if candidate_id in kept_melodic_ids:
                replacement_inst_id = candidate_id
                break
        if replacement_inst_id is None:
            replacement_inst_id = fallback_inst_id

        remapped_notes.append(replace(note, instrument_id=int(replacement_inst_id)))
        remapped_note_count += 1

    return sorted(
        remapped_notes,
        key=lambda note: (
            note.start_frame,
            note.pitch,
            note.instrument_id,
            note.end_frame,
        ),
    ), {
        "midi_instrument_count_before_remap": len(counts),
        "midi_instrument_count_after_remap": len(
            {int(note.instrument_id) for note in remapped_notes}
        ),
        "remapped_instrument_count": len(overflow_ids),
        "remapped_note_count": remapped_note_count,
    }


def _build_midi(
    notes: list[PredictedNote],
    *,
    sample_rate: int,
) -> pretty_midi.PrettyMIDI:
    from instrument_classes import INSTRUMENT_CLASSES, get_program_number_from_class_id

    midi = pretty_midi.PrettyMIDI()

    notes_by_inst: dict[int, list[PredictedNote]] = {}
    for note in notes:
        notes_by_inst.setdefault(note.instrument_id, []).append(note)

    for inst_id, inst_notes in notes_by_inst.items():
        class_name = (
            INSTRUMENT_CLASSES[inst_id]
            if 0 <= inst_id < len(INSTRUMENT_CLASSES)
            else "Piano"
        )
        prog_num = get_program_number_from_class_id(inst_id)
        is_drum = class_name.lower() == "drums"

        instrument = pretty_midi.Instrument(
            program=prog_num, is_drum=is_drum, name=class_name
        )
        inst_notes = _truncate_overlapping_notes(
            inst_notes,
            separate_adjacent_same_pitch=True,
            min_separation_frames=max(1, int(round(float(sample_rate) * 0.005))),
        )
        for note in inst_notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(note.velocity),
                    pitch=int(note.pitch),
                    start=float(note.start_frame) / float(sample_rate),
                    end=float(note.end_frame) / float(sample_rate),
                )
            )
        midi.instruments.append(instrument)

    return midi


@torch.inference_mode()
def run_inference(
    *,
    model: AudioSemiCRFTransformer,
    waveform: torch.Tensor,
    model_config: SemiCRFModelConfig,
    settings: InferenceSettings,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    velocity: int,
    merge_gap_ms: float | None,
    merge_onset_ms: float,
    silence_gate_rms_dbfs: float | None,
    window_batch_size: int,
    max_midi_melodic_instruments: int,
    disable_tqdm: bool,
) -> tuple[list[PredictedNote], dict[str, int]]:
    if waveform.dim() != 2 or int(waveform.shape[0]) != 2:
        raise ValueError("waveform must have shape [2, audio_frames]")

    sample_rate = int(model_config.sample_rate)
    total_audio_frames = int(waveform.shape[-1])
    window_audio_frames = int(round(settings.window_ms * sample_rate / 1000.0))
    stride_audio_frames = int(round(settings.stride_ms * sample_rate / 1000.0))
    if window_audio_frames < int(model_config.n_fft):
        raise ValueError(
            f"window_ms={settings.window_ms} is too short for n_fft={model_config.n_fft}"
        )
    if total_audio_frames < int(model_config.n_fft):
        raise ValueError(
            f"Audio is too short for n_fft={model_config.n_fft}: {total_audio_frames} frames"
        )
    if stride_audio_frames <= 0:
        raise ValueError("stride_ms must be positive")

    window_starts = _build_window_starts(
        total_audio_frames=total_audio_frames,
        window_audio_frames=window_audio_frames,
        stride_audio_frames=stride_audio_frames,
    )
    silence_gate_linear = _silence_gate_rms_linear(silence_gate_rms_dbfs)
    predicted_notes: list[PredictedNote] = []
    skipped_silent_window_count = 0
    decoded_window_count = 0
    progress = tqdm(
        _iter_batches(window_starts, window_batch_size),
        total=math.ceil(len(window_starts) / max(1, int(window_batch_size))),
        desc="infer",
        dynamic_ncols=True,
        disable=bool(disable_tqdm),
    )

    use_boundary_stitching = bool(model.supports_interval_boundaries())
    for batch_starts in progress:
        window_tensors = []
        valid_audio_frames = []
        for start_frame in batch_starts:
            window, valid_frames = _slice_window(
                waveform,
                start_frame=int(start_frame),
                window_audio_frames=window_audio_frames,
            )
            window_tensors.append(window)
            valid_audio_frames.append(int(valid_frames))

        batch_waveform_cpu = torch.stack(window_tensors, dim=0)
        silent_window_mask = _compute_silent_window_mask(
            batch_waveform_cpu,
            silence_gate_rms_linear=silence_gate_linear,
        )
        if silent_window_mask is not None:
            skipped_silent_window_count += int(silent_window_mask.sum().item())
            active_indices = [
                index
                for index, is_silent in enumerate(silent_window_mask.tolist())
                if not is_silent
            ]
        else:
            active_indices = list(range(len(batch_starts)))

        if not active_indices:
            continue

        active_batch_starts = [int(batch_starts[index]) for index in active_indices]
        active_valid_audio_frames = [
            int(valid_audio_frames[index]) for index in active_indices
        ]
        decoded_window_count += len(active_indices)

        batch_waveform = batch_waveform_cpu[active_indices].to(device)
        valid_audio_frames_tensor = torch.tensor(
            active_valid_audio_frames,
            dtype=torch.long,
            device=device,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_enabled and device.type == "cuda"),
        ):
            outputs = model(
                batch_waveform,
                valid_audio_frames=valid_audio_frames_tensor,
            )

        valid_lengths = outputs["frame_valid_mask"].to(dtype=torch.long).sum(dim=-1)
        decoded_intervals = decode_pitch_intervals(
            outputs["interval_query"],
            outputs["interval_key"],
            outputs["interval_diag"],
            valid_lengths,
            length_scaling=settings.length_scaling,
            length_penalty=settings.length_penalty,
            track_batch_size=settings.track_batch_size,
        )

        boundary_flags_batch: (
            list[list[list[tuple[bool, bool, float, float]]]] | None
        ) = None
        if use_boundary_stitching:
            boundary_logits, boundary_entries = model.predict_interval_boundaries(
                outputs.get("interval_features", outputs["pitch_query_features"]),
                decoded_intervals,
            )
            boundary_flags_batch = _decode_boundary_features(
                boundary_logits,
                boundary_entries,
                batch_size=len(active_batch_starts),
                num_pitches=int(outputs["interval_query"].shape[2]),
            )

        sample_logits_batch = outputs.get("instrument_logits")

        for sample_index, (
            start_frame,
            valid_frames,
            intervals_by_pitch,
        ) in enumerate(
            zip(active_batch_starts, active_valid_audio_frames, decoded_intervals)
        ):
            sample_boundary_flags = (
                boundary_flags_batch[sample_index]
                if boundary_flags_batch is not None
                else None
            )
            sample_logits = (
                sample_logits_batch[sample_index, :valid_frames]
                if sample_logits_batch is not None
                else None
            )
            predicted_notes.extend(
                _intervals_to_notes(
                    intervals_by_pitch,
                    boundary_flags_by_pitch=sample_boundary_flags,
                    instrument_logits=sample_logits,
                    window_start_frame=int(start_frame),
                    valid_audio_frames=int(valid_frames),
                    total_audio_frames=total_audio_frames,
                    hop_length=int(model_config.hop_length),
                    velocity=int(velocity),
                )
            )

    merge_gap_frames = (
        int(model_config.hop_length)
        if merge_gap_ms is None
        else max(0, int(round(float(merge_gap_ms) * sample_rate / 1000.0)))
    )
    merge_onset_frames = max(
        0, int(round(float(merge_onset_ms) * sample_rate / 1000.0))
    )

    # 1. 境界線を利用したステッチング（利用可能な場合）
    notes_after_stitching = (
        _stitch_boundary_aware_notes(predicted_notes)
        if use_boundary_stitching
        else predicted_notes
    )

    # 2. 重複・近接ノートの最終的なマージ処理（オンセット距離も考慮）
    merged_notes = _merge_notes(
        notes_after_stitching,
        merge_gap_frames=merge_gap_frames,
        merge_onset_frames=merge_onset_frames,
    )

    remapped_notes, remap_stats = _remap_overflow_midi_instruments(
        merged_notes,
        max_melodic_instruments=int(max_midi_melodic_instruments),
    )
    if remap_stats["remapped_note_count"] > 0:
        merged_notes = _truncate_overlapping_notes(
            remapped_notes,
            separate_adjacent_same_pitch=True,
            min_separation_frames=max(1, int(round(float(sample_rate) * 0.005))),
        )
        remap_stats["midi_instrument_count_after_remap"] = len(
            {int(note.instrument_id) for note in merged_notes}
        )
    else:
        merged_notes = _truncate_overlapping_notes(
            remapped_notes,
            separate_adjacent_same_pitch=True,
            min_separation_frames=max(1, int(round(float(sample_rate) * 0.005))),
        )

    return merged_notes, {
        "window_count": len(window_starts),
        "decoded_window_count": int(decoded_window_count),
        "skipped_silent_window_count": int(skipped_silent_window_count),
        "window_audio_frames": int(window_audio_frames),
        "stride_audio_frames": int(stride_audio_frames),
        "merge_gap_frames": int(merge_gap_frames),
        "merge_onset_frames": int(merge_onset_frames),
        **remap_stats,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    audio_path = args.audio.resolve()
    output_midi_path = (
        args.output_midi.resolve()
        if args.output_midi is not None
        else audio_path.with_suffix(".mid")
    )

    checkpoint_path = _ensure_checkpoint(args.checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}...")
    model, model_config, settings = _load_model_and_settings(
        checkpoint_path.resolve(),
        device=device,
        window_ms_override=args.window_ms,
        stride_ms_override=args.stride_ms,
        track_batch_size_override=args.semi_crf_track_batch_size,
    )

    print(f"Loading audio from {audio_path}...")
    waveform, source_sample_rate, source_channels = _load_audio(
        audio_path,
        target_sample_rate=int(model_config.sample_rate),
    )

    print("Running inference...")
    notes, stats = run_inference(
        model=model,
        waveform=waveform,
        model_config=model_config,
        settings=settings,
        device=device,
        amp_enabled=bool(args.amp),
        amp_dtype=amp_dtype,
        velocity=int(args.velocity),
        merge_gap_ms=args.merge_gap_ms,
        merge_onset_ms=args.merge_onset_ms,
        silence_gate_rms_dbfs=args.silence_gate_rms_dbfs,
        window_batch_size=int(args.window_batch_size),
        max_midi_melodic_instruments=int(args.max_midi_melodic_instruments),
        disable_tqdm=bool(args.disable_tqdm),
    )

    midi = _build_midi(notes, sample_rate=int(model_config.sample_rate))
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_midi_path))

    duration_seconds = float(waveform.shape[-1]) / float(model_config.sample_rate)
    print(
        f"audio_source_sr={source_sample_rate} "
        f"model_sr={model_config.sample_rate} "
        f"source_channels={source_channels} "
        f"used_channels=2"
    )
    print(
        f"duration_seconds={duration_seconds:.2f} "
        f"window_ms={settings.window_ms} "
        f"stride_ms={settings.stride_ms} "
        f"windows={stats['window_count']} "
        f"decoded_windows={stats['decoded_window_count']} "
        f"skipped_silent_windows={stats['skipped_silent_window_count']}"
    )
    print(
        f"decoded_notes={len(notes)} "
        f"midi_instruments_before_remap={stats['midi_instrument_count_before_remap']} "
        f"midi_instruments_after_remap={stats['midi_instrument_count_after_remap']} "
        f"remapped_instruments={stats['remapped_instrument_count']} "
        f"remapped_notes={stats['remapped_note_count']} "
        f"silence_gate_rms_dbfs={args.silence_gate_rms_dbfs} "
        f"output_midi={output_midi_path}"
    )


if __name__ == "__main__":
    main()
