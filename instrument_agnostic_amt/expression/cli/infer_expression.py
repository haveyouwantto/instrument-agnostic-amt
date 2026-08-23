"""Experimental per-stem expression (CC11) estimation.

For each stem, the *entire* audio is analyzed with an STFT (flattop window,
2048 FFT / 512 hop, the same calculation used by ``velocity_estimator.py``)
and per-frame magnitudes are converted to dB.  For sustained / orchestral
instruments only, CC events are written at millisecond resolution (default
20 ms grid) while the instrument is sounding, following the per-frame dB
curve mapped with the original STFT range: 0..48 dB -> 8..127 with a
power-law curve (exponent 1.2), then normalized/stretched per track.

Why this merge strategy:
- Per-note CC values on the same track fight when notes overlap or when the
  same tick receives multiple events.  We therefore deduplicate by tick
  (later frame wins) and collapse consecutive equal values, so every track
  gets a dense, conflict-free expression curve.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal.windows import flattop


# Default GM program ranges treated as sustained / orchestral.
# (lo, hi) with lo <= program < hi.
SUSTAINED_PROGRAM_RANGES: tuple[tuple[int, int], ...] = (
    (16, 24),    # organ
    (40, 46),    # violin / viola / cello / contrabass / tremolo / pizzicato
    (48, 50),    # string ensemble 1/2
    (52, 54),    # choir aahs / voice oohs
    (56, 80),    # brass + reeds / woodwinds
)


@dataclass
class _Segment:
    """A merged phrase interval (start, end) and its target CC value."""

    start: float
    end: float
    cc: int


def _resolve_audio_paths(
    midi_path: Path,
    stem_name: str,
    audio_sources: Sequence[Path],
) -> Path | None:
    """Match a stem MIDI to its audio file inside the pipeline output layout."""
    midi_stem = midi_path.stem
    candidates: list[Path] = []
    for audio_path in audio_sources:
        audio_stem = audio_path.stem
        if audio_stem == midi_stem:
            return audio_path
        if stem_name and (audio_stem.endswith(f"_{stem_name}") or audio_stem == stem_name):
            candidates.append(audio_path)
        elif stem_name and f"_{stem_name}." in audio_path.name:
            candidates.append(audio_path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: len(path.name))
    return candidates[0]


def _list_audio_files(audio_source: Path) -> list[Path]:
    if not audio_source.exists():
        return []
    if audio_source.is_file():
        return [audio_source] if audio_source.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg") else []
    return sorted(
        path
        for path in audio_source.rglob("*")
        if path.is_file() and path.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg")
    )


def _stft_peak_magnitude_db(
    mono: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Per-frame peak-bin STFT magnitude in dB (flattop window, raw rfft)."""
    if mono.size < n_fft:
        mono = np.pad(mono, (0, n_fft - int(mono.size)))
    num_frames = 1 + (int(mono.size) - n_fft) // hop_length
    window = flattop(n_fft).astype(np.float64)
    db_values = np.empty(num_frames, dtype=np.float64)
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        spectrum = np.abs(np.fft.rfft(mono[start : start + n_fft] * window))
        peak_magnitude = float(np.max(spectrum))
        db_values[frame_idx] = 20.0 * np.log10(max(peak_magnitude, 1e-10))
    return db_values


def estimate_loudness_curve(
    audio_path: Path | str,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    smoothing_seconds: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the whole-stem STFT loudness envelope in dB."""
    waveform, source_sr = sf.read(
        str(audio_path), dtype="float32", always_2d=True
    )
    if waveform.shape[1] > 2:
        waveform = waveform[:, :2]
    mono = waveform.mean(axis=1).astype(np.float64)
    db_values = _stft_peak_magnitude_db(
        mono,
        int(source_sr),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
    )
    kernel = max(
        1,
        int(round(float(smoothing_seconds) * int(source_sr) / float(hop_length))),
    )
    if db_values.shape[0] >= kernel:
        padded = np.pad(db_values, (kernel // 2, kernel - 1 - kernel // 2), mode="edge")
        kernel_array = np.ones(kernel, dtype=np.float64) / float(kernel)
        db_values = np.convolve(padded, kernel_array, mode="valid")
    times = np.arange(int(db_values.shape[0]), dtype=np.float64) * float(hop_length) / float(source_sr)
    return times, db_values


def _pitch_to_frequency(pitch: int, *, a4_freq: float = 440.0) -> float:
    return float(a4_freq) * 2.0 ** ((int(pitch) - 69) / 12.0)


def _pitch_to_stft_bin(
    pitch: int,
    sample_rate: int,
    n_fft: int,
) -> int:
    frequency = _pitch_to_frequency(pitch)
    return int(round(frequency * int(n_fft) / int(sample_rate)))


def _stft_magnitude_spectrogram(
    mono: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, int]:
    """Full magnitude spectrogram [frames, n_fft//2+1] with the flattop window."""
    if mono.size < n_fft:
        mono = np.pad(mono, (0, n_fft - int(mono.size)))
    num_frames = 1 + (int(mono.size) - n_fft) // hop_length
    window = flattop(n_fft).astype(np.float64)
    spectrogram = np.empty(
        (num_frames, n_fft // 2 + 1),
        dtype=np.float32,
    )
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        spectrum = np.abs(np.fft.rfft(mono[start : start + n_fft] * window))
        spectrogram[frame_idx, :] = spectrum.astype(np.float32)
    return spectrogram, num_frames


def _spectrogram_from_audio(
    audio_path: Path | str,
    *,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, int]:
    waveform, source_sr = sf.read(
        str(audio_path), dtype="float32", always_2d=True
    )
    if waveform.shape[1] > 2:
        waveform = waveform[:, :2]
    mono = waveform.mean(axis=1).astype(np.float64)
    spectrogram, _num_frames = _stft_magnitude_spectrogram(
        mono,
        int(source_sr),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
    )
    return spectrogram, int(source_sr)


def _frame_range(
    start_seconds: float,
    end_seconds: float,
    frame_times: np.ndarray,
) -> tuple[int, int]:
    start_frame = int(np.searchsorted(frame_times, start_seconds, side="left"))
    end_frame = int(np.searchsorted(frame_times, end_seconds, side="right")) - 1
    start_frame = max(0, start_frame)
    end_frame = min(int(frame_times.shape[0]) - 1, end_frame)
    if end_frame < start_frame:
        return -1, -1
    return start_frame, end_frame


def instrument_curve_from_spectrogram(
    spectrogram: np.ndarray,
    sample_rate: int,
    notes: Sequence[pretty_midi.Note],
    *,
    n_fft: int,
    hop_length: int,
    smoothing_seconds: float = 0.1,
    min_active_db: float = -60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-instrument dB curve from its own notes' fundamental energy.

    For each STFT frame, the strongest magnitude among the fundamental bins
    of the notes that are sounding at that moment is used, so every instrument
    receives its own loudness curve instead of sharing the stem-level one.
    Frames where the instrument has no active note are dropped.
    """
    num_frames = int(spectrogram.shape[0])
    if num_frames == 0 or not notes:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    frame_times = (
        np.arange(num_frames, dtype=np.float64) * float(hop_length) / float(sample_rate)
    )
    db_values = np.full(num_frames, -120.0, dtype=np.float64)

    for note in notes:
        start_frame, end_frame = _frame_range(
            float(note.start),
            float(note.end),
            frame_times,
        )
        if start_frame < 0:
            continue
        bin_index = _pitch_to_stft_bin(int(note.pitch), int(sample_rate), int(n_fft))
        low_bin = max(1, bin_index - 1)
        high_bin = min(int(spectrogram.shape[1]) - 1, bin_index + 2)
        for frame_idx in range(start_frame, end_frame + 1):
            magnitude = float(
                np.max(spectrogram[frame_idx, low_bin:high_bin])
            )
            frame_db = 20.0 * np.log10(max(magnitude, 1e-10))
            if frame_db > db_values[frame_idx]:
                db_values[frame_idx] = frame_db

    kernel = max(
        1,
        int(round(float(smoothing_seconds) * int(sample_rate) / float(hop_length))),
    )
    if db_values.shape[0] >= kernel:
        padded = np.pad(
            db_values,
            (kernel // 2, kernel - 1 - kernel // 2),
            mode="edge",
        )
        kernel_array = np.ones(kernel, dtype=np.float64) / float(kernel)
        db_values = np.convolve(padded, kernel_array, mode="valid")

    active = db_values > float(min_active_db)
    return frame_times[active], db_values[active]


def estimate_instrument_loudness_curve(
    audio_path: Path | str,
    notes: Sequence[pretty_midi.Note],
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    smoothing_seconds: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: spectrogram + per-instrument curve for one track."""
    spectrogram, sample_rate = _spectrogram_from_audio(
        audio_path,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
    )
    return instrument_curve_from_spectrogram(
        spectrogram,
        sample_rate,
        notes,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        smoothing_seconds=smoothing_seconds,
    )


def _db_to_cc(
    db_value: float,
    *,
    min_db: float = 0.0,
    max_db: float = 48.0,
    cc_min: int = 8,
    cc_max: int = 127,
    curve_exponent: float = 1.2,
) -> int:
    """Map STFT-magnitude dB to CC with the original velocity-estimator curve."""
    clamped = np.clip(float(db_value), float(min_db), float(max_db))
    normalized = (clamped - float(min_db)) / (float(max_db) - float(min_db))
    curved = normalized ** float(curve_exponent)
    cc_value = int(round(float(cc_min) + curved * float(cc_max - cc_min)))
    return max(int(cc_min), min(int(cc_max), cc_value))


def _stretch_cc_values(
    values: Sequence[int],
    *,
    cc_min: int,
    cc_max: int,
    dynamic_stretch: float,
) -> list[int]:
    """Normalize the peak to ``cc_max`` and stretch dynamics below it."""
    if not values:
        return []
    peak = max(values)
    if peak <= 0:
        return [int(cc_min)] * len(values)
    stretched: list[int] = []
    for value in values:
        normalized = float(np.clip(float(value) * float(cc_max) / float(peak), cc_min, cc_max))
        if dynamic_stretch > 1.0:
            normalized = float(cc_max) - (
                float(cc_max) - normalized
            ) * float(dynamic_stretch)
        stretched.append(
            int(round(float(np.clip(normalized, cc_min, cc_max))))
        )
    return stretched


def build_frame_cc_events(
    midi: pretty_midi.PrettyMIDI,
    curve_times: np.ndarray,
    curve_values: np.ndarray,
    notes: Sequence[pretty_midi.Note],
    *,
    cc: int = 11,
    interval_seconds: float = 0.02,
    cc_min: int,
    cc_max: int,
    dynamic_stretch: float = 1.0,
    min_db: float = 0.0,
    max_db: float = 48.0,
    curve_exponent: float = 1.2,
) -> list[tuple[float, int]]:
    """Build a dense, millisecond-resolution CC event list for one track.

    Every curve frame inside the track's note span (first note start .. last
    note end) is mapped to a CC value via the 0..48 dB -> 8..127 power-law
    curve, then normalized/stretched per track.  Events are written on a
    ``interval_seconds`` grid (default 20 ms), deduplicated by tick (later
    frame wins) and consecutive equal values are collapsed, so the track gets
    a continuously-moving expression curve without same-tick conflicts.
    """
    if not notes:
        return []
    note_start = min(float(note.start) for note in notes)
    note_end = max(float(note.end) for note in notes)
    mask = (curve_times >= note_start) & (curve_times <= note_end)
    frame_times = curve_times[mask]
    frame_db = curve_values[mask]
    if frame_times.shape[0] == 0:
        return []

    frame_cc = [
        _db_to_cc(
            float(db),
            min_db=min_db,
            max_db=max_db,
            cc_min=cc_min,
            cc_max=cc_max,
            curve_exponent=curve_exponent,
        )
        for db in frame_db
    ]
    frame_cc = _stretch_cc_values(
        frame_cc,
        cc_min=cc_min,
        cc_max=cc_max,
        dynamic_stretch=dynamic_stretch,
    )

    # Re-sample onto the requested millisecond grid, dedupe by tick (later
    # frame wins), then collapse consecutive equal values.
    frame_dt = (
        float(curve_times[1] - curve_times[0])
        if curve_times.shape[0] > 1
        else 0.0
    )
    step = (
        max(1, int(round(float(interval_seconds) / frame_dt)))
        if frame_dt > 0
        else 1
    )
    by_tick: dict[int, int] = {}
    for frame_index in range(0, int(frame_times.shape[0]), step):
        tick = int(midi.time_to_tick(float(frame_times[frame_index])))
        by_tick[tick] = int(frame_cc[frame_index])

    events: list[tuple[float, int]] = []
    last_value: int | None = None
    for tick in sorted(by_tick):
        value = by_tick[tick]
        if value == last_value:
            continue
        events.append((float(midi.tick_to_time(tick)), value))
        last_value = value
    return events


def _is_sustained(
    program: int,
    is_drum: bool,
    ranges: Sequence[tuple[int, int]],
) -> bool:
    if is_drum:
        return False
    return any(lower <= int(program) < upper for lower, upper in ranges)


def apply_expression_to_midi(
    midi: pretty_midi.PrettyMIDI,
    curve_times: np.ndarray,
    curve_values: np.ndarray,
    *,
    cc: int = 11,
    sustained_ranges: Sequence[tuple[int, int]] = SUSTAINED_PROGRAM_RANGES,
    interval_seconds: float = 0.02,
    cc_min: int = 8,
    cc_max: int = 127,
    dynamic_stretch: float = 1.0,
    smoothing_seconds: float = 0.1,
    min_db: float = 0.0,
    max_db: float = 48.0,
    curve_exponent: float = 1.2,
) -> int:
    """Write CC events onto sustained instruments of ``midi`` in place.

    Returns the number of instruments that received CC events.  Existing CC
    events with the same controller number are replaced, and events are
    deduplicated by tick so no two CC messages share a tick on one track.
    """
    changed_instruments = 0
    for instrument in midi.instruments:
        if _apply_expression_to_instrument(
            instrument,
            curve_times,
            curve_values,
            cc=cc,
            sustained_ranges=sustained_ranges,
            interval_seconds=interval_seconds,
            cc_min=cc_min,
            cc_max=cc_max,
            dynamic_stretch=dynamic_stretch,
            min_db=min_db,
            max_db=max_db,
            curve_exponent=curve_exponent,
            midi=midi,
        ):
            changed_instruments += 1
    return changed_instruments


def _apply_expression_to_instrument(
    instrument: pretty_midi.Instrument,
    curve_times: np.ndarray,
    curve_values: np.ndarray,
    *,
    cc: int,
    sustained_ranges: Sequence[tuple[int, int]],
    interval_seconds: float,
    cc_min: int,
    cc_max: int,
    dynamic_stretch: float,
    min_db: float,
    max_db: float,
    curve_exponent: float,
    midi: pretty_midi.PrettyMIDI,
) -> bool:
    """Write CC events onto one sustained instrument using the given curve."""
    if not _is_sustained(int(instrument.program), bool(instrument.is_drum), sustained_ranges):
        return False
    if not instrument.notes:
        return False
    instrument.control_changes = [
        control
        for control in instrument.control_changes
        if int(control.number) != int(cc)
    ]
    events = build_frame_cc_events(
        midi,
        curve_times,
        curve_values,
        instrument.notes,
        cc=cc,
        interval_seconds=interval_seconds,
        cc_min=cc_min,
        cc_max=cc_max,
        dynamic_stretch=dynamic_stretch,
        min_db=min_db,
        max_db=max_db,
        curve_exponent=curve_exponent,
    )
    if not events:
        return False
    for start, value in events:
        instrument.control_changes.append(
            pretty_midi.ControlChange(
                number=int(cc),
                value=int(value),
                time=float(start),
            )
        )
    instrument.control_changes.sort(key=lambda control: control.time)
    return True


def apply_expression_to_merged_midi(
    merged_midi_path: Path | str,
    stem_midis: Mapping[str, Path | str] | Sequence[Path | str],
    stem_audios: Mapping[str, Path | str] | Path | str,
    *,
    output_midi_path: Path | str | None = None,
    cc: int = 11,
    sustained_ranges: Sequence[tuple[int, int]] = SUSTAINED_PROGRAM_RANGES,
    n_fft: int = 2048,
    hop_length: int = 512,
    interval_seconds: float = 0.02,
    cc_min: int = 8,
    cc_max: int = 127,
    dynamic_stretch: float = 1.0,
    smoothing_seconds: float = 0.1,
    min_db: float = 0.0,
    max_db: float = 48.0,
    curve_exponent: float = 1.2,
) -> Path:
    """Write CC expression curves onto the merged MIDI, keeping its tempo map.

    Each sustained instrument of the merged file is mapped back to the stem
    that produced it (by program / drum flag / instrument name) and gets its
    own loudness curve by tracking the fundamental energy of its own notes in
    that stem's STFT spectrogram.
    """
    merged_path = Path(merged_midi_path)
    midi_obj = pretty_midi.PrettyMIDI(str(merged_path))

    if isinstance(stem_midis, Mapping):
        stem_midi_items = list(stem_midis.items())
    else:
        stem_midi_items = [(Path(path).stem, Path(path)) for path in stem_midis]

    audio_by_name: dict[str, Path] = {}
    if isinstance(stem_audios, Mapping):
        for name, path in stem_audios.items():
            audio_by_name[str(name).lower()] = Path(path)
    else:
        audio_files = _list_audio_files(Path(stem_audios))
        for name, path in stem_midi_items:
            audio_by_name[str(name).lower()] = _resolve_audio_paths(
                Path(path), str(name), audio_files
            )

    stem_instrument_keys: dict[str, set[tuple[int, bool, str]]] = {}
    for stem_name, midi_path in stem_midi_items:
        stem_midi = Path(midi_path)
        if not stem_midi.exists():
            continue
        pm_obj = pretty_midi.PrettyMIDI(str(stem_midi))
        stem_instrument_keys[stem_name] = {
            (int(inst.program), bool(inst.is_drum), str(inst.name))
            for inst in pm_obj.instruments
        }

    spectrogram_cache: dict[str, tuple[np.ndarray, int]] = {}
    changed = 0
    for instrument in midi_obj.instruments:
        key = (int(instrument.program), bool(instrument.is_drum), str(instrument.name))
        stem_name = next(
            (
                name
                for name, keys in stem_instrument_keys.items()
                if key in keys
            ),
            None,
        )
        if stem_name is None:
            continue
        audio_path = audio_by_name.get(str(stem_name).lower())
        if audio_path is None or not audio_path.exists():
            print(
                f"[Expression] WARNING: no audio for stem {stem_name}, "
                f"skipping {instrument.name}"
            )
            continue
        if stem_name not in spectrogram_cache:
            spectrogram_cache[stem_name] = _spectrogram_from_audio(
                audio_path,
                n_fft=n_fft,
                hop_length=hop_length,
            )
        spectrogram, sample_rate = spectrogram_cache[stem_name]
        curve_times, curve_values = instrument_curve_from_spectrogram(
            spectrogram,
            sample_rate,
            instrument.notes,
            n_fft=n_fft,
            hop_length=hop_length,
            smoothing_seconds=smoothing_seconds,
        )
        if curve_times.shape[0] == 0:
            continue
        if _apply_expression_to_instrument(
            instrument,
            curve_times,
            curve_values,
            cc=cc,
            sustained_ranges=sustained_ranges,
            interval_seconds=interval_seconds,
            cc_min=cc_min,
            cc_max=cc_max,
            dynamic_stretch=dynamic_stretch,
            min_db=min_db,
            max_db=max_db,
            curve_exponent=curve_exponent,
            midi=midi_obj,
        ):
            changed += 1

    output_path = Path(output_midi_path) if output_midi_path is not None else merged_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_obj.write(str(output_path))
    print(f"[Expression] {changed} sustained instrument(s) on merged MIDI received CC{cc}")
    return output_path


def predict_expression_for_stem_midis(
    stem_midis: Mapping[str, Path | str] | Sequence[Path | str],
    stem_audios: Path | str | Mapping[str, Path | str] | None = None,
    *,
    output_dir: Path | str | None = None,
    in_place: bool = False,
    cc: int = 11,
    sustained_ranges: Sequence[tuple[int, int]] = SUSTAINED_PROGRAM_RANGES,
    n_fft: int = 2048,
    hop_length: int = 512,
    interval_seconds: float = 0.02,
    cc_min: int = 8,
    cc_max: int = 127,
    dynamic_stretch: float = 1.0,
    smoothing_seconds: float = 0.1,
    min_db: float = 0.0,
    max_db: float = 48.0,
    curve_exponent: float = 1.2,
    quiet: bool = False,
) -> dict[str, Path]:
    """Apply per-instrument CC expression curves to each stem MIDI.

    Every sustained instrument tracks the fundamental energy of its own notes
    in the stem audio's STFT spectrogram, so each track gets its own curve.
    Returns a mapping of stem name -> output MIDI path.
    """
    if isinstance(stem_midis, Mapping):
        midi_items = list(stem_midis.items())
    else:
        midi_items = [(Path(path).stem, Path(path)) for path in stem_midis]

    audio_by_name: dict[str, Path] = {}
    if isinstance(stem_audios, Mapping):
        for name, path in stem_audios.items():
            audio_by_name[str(name).lower()] = Path(path)
        audio_files: list[Path] = []
    else:
        audio_files = (
            _list_audio_files(Path(stem_audios))
            if stem_audios is not None
            else []
        )

    resolved_output: dict[str, Path] = {}
    for stem_name, midi_path_item in midi_items:
        midi_path = Path(midi_path_item)
        if not midi_path.exists():
            continue
        audio_path = audio_by_name.get(str(stem_name).lower())
        if audio_path is None:
            audio_path = _resolve_audio_paths(midi_path, str(stem_name), audio_files)
        if audio_path is None:
            print(f"[Expression] WARNING: no audio found for {midi_path.name}, skipping")
            continue

        spectrogram, sample_rate = _spectrogram_from_audio(
            audio_path,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        midi_obj = pretty_midi.PrettyMIDI(str(midi_path))
        changed = 0
        for instrument in midi_obj.instruments:
            if not _is_sustained(
                int(instrument.program),
                bool(instrument.is_drum),
                sustained_ranges,
            ):
                continue
            if not instrument.notes:
                continue
            curve_times, curve_values = instrument_curve_from_spectrogram(
                spectrogram,
                sample_rate,
                instrument.notes,
                n_fft=n_fft,
                hop_length=hop_length,
                smoothing_seconds=smoothing_seconds,
            )
            if curve_times.shape[0] == 0:
                continue
            if _apply_expression_to_instrument(
                instrument,
                curve_times,
                curve_values,
                cc=cc,
                sustained_ranges=sustained_ranges,
                interval_seconds=interval_seconds,
                cc_min=cc_min,
                cc_max=cc_max,
                dynamic_stretch=dynamic_stretch,
                min_db=min_db,
                max_db=max_db,
                curve_exponent=curve_exponent,
                midi=midi_obj,
            ):
                changed += 1
        if in_place:
            output_path = midi_path
        else:
            output_dir_path = Path(output_dir) if output_dir is not None else midi_path.parent
            output_dir_path.mkdir(parents=True, exist_ok=True)
            output_path = output_dir_path / f"{midi_path.stem}_expression.mid"
        midi_obj.write(str(output_path))
        resolved_output[str(stem_name)] = output_path
        if not quiet:
            print(
                f"[Expression] {midi_path.name}: {changed} sustained instrument(s) "
                f"received CC{cc} from {audio_path.name}"
            )
    return resolved_output


def merge_expression_midis(
    midi_paths: Sequence[Path | str],
    output_file: Path | str,
) -> Path:
    """Merge per-stem MIDIs into one file, keyed by (program, drum, name)."""
    output_path = Path(output_file)
    master = pretty_midi.PrettyMIDI(str(midi_paths[0]))
    grouped: dict[tuple[int, bool, str], pretty_midi.Instrument] = {}
    for path in midi_paths:
        midi_obj = pretty_midi.PrettyMIDI(str(path))
        for instrument in midi_obj.instruments:
            key = (int(instrument.program), bool(instrument.is_drum), str(instrument.name))
            target = grouped.get(key)
            if target is None:
                target = pretty_midi.Instrument(
                    program=int(instrument.program),
                    is_drum=bool(instrument.is_drum),
                    name=str(instrument.name),
                )
                grouped[key] = target
            target.notes.extend(instrument.notes)
            target.control_changes.extend(instrument.control_changes)
            target.pitch_bends.extend(instrument.pitch_bends)
    for instrument in grouped.values():
        instrument.notes.sort(key=lambda note: note.start)
        instrument.control_changes.sort(key=lambda control: control.time)
    master.instruments = list(grouped.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.write(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-stem volume curves from the full audio and write "
            "CC11 expression events onto sustained (orchestral) instruments."
        )
    )
    parser.add_argument(
        "song_dir",
        type=Path,
        help=(
            "Pipeline output directory containing stem_midis/*.mid and "
            "stems/ (wav) subdirectories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write *_expression.mid files (default: alongside inputs).",
    )
    parser.add_argument("--cc", type=int, default=11, help="Controller number (default: 11).")
    parser.add_argument("--n-fft", type=int, default=2048, help="STFT window size (samples).")
    parser.add_argument("--hop-length", type=int, default=512, help="STFT hop length (samples).")
    parser.add_argument(
        "--cc-interval-ms",
        type=float,
        default=20.0,
        help="CC event grid interval in milliseconds (default 20 ms).",
    )
    parser.add_argument(
        "--smoothing-seconds",
        type=float,
        default=0.1,
        help="Per-instrument dB curve smoothing in seconds.",
    )
    parser.add_argument("--min-db", type=float, default=0.0, help="Minimum dB for the CC mapping.")
    parser.add_argument("--max-db", type=float, default=48.0, help="Maximum dB for the CC mapping.")
    parser.add_argument(
        "--curve-exponent",
        type=float,
        default=1.2,
        help="Power-law curve exponent for dB -> CC.",
    )
    parser.add_argument("--cc-min", type=int, default=8, help="Minimum CC value.")
    parser.add_argument("--cc-max", type=int, default=127, help="Maximum CC value.")
    parser.add_argument(
        "--dynamic-stretch",
        type=float,
        default=1.0,
        help="Normalize each track's peak to CC max and stretch its dynamics by this factor.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also merge the expression MIDIs into one file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    midi_dir = args.song_dir / "stem_midis"
    audio_dir = args.song_dir / "stems"
    midi_paths = sorted(midi_dir.glob("*.mid")) if midi_dir.exists() else []
    if not midi_paths:
        raise SystemExit(f"No stem MIDI files found under {midi_dir}")
    output_dir = args.output_dir or args.song_dir / "stem_midis_expression"
    outputs = predict_expression_for_stem_midis(
        midi_paths,
        stem_audios=audio_dir,
        output_dir=output_dir,
        cc=args.cc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        interval_seconds=args.cc_interval_ms / 1000.0,
        smoothing_seconds=args.smoothing_seconds,
        cc_min=args.cc_min,
        cc_max=args.cc_max,
        dynamic_stretch=args.dynamic_stretch,
        min_db=args.min_db,
        max_db=args.max_db,
        curve_exponent=args.curve_exponent,
    )
    if not outputs:
        raise SystemExit("No expression MIDI files were produced.")
    if args.merge and outputs:
        merged_path = args.output_dir / f"{args.song_dir.name}_expression.mid" if args.output_dir else (
            args.song_dir / "merged" / f"{args.song_dir.name}_expression.mid"
        )
        merged = merge_expression_midis(list(outputs.values()), merged_path)
        print(f"[Expression] Merged -> {merged}")


if __name__ == "__main__":
    main()
