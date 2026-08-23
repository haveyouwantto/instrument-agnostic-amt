"""Experimental per-stem expression (CC11) estimation.

For each stem, the *entire* audio is analyzed to estimate a loudness envelope.
For sustained / orchestral instruments only, the transcribed notes are grouped
into phrase-like segments and each segment receives a single CC event whose
value is derived from the loudness envelope of that time interval.

Why this merge strategy:
- Per-note CC values on the same track fight when notes overlap or when the
  same tick receives multiple events.  We therefore emit at most one CC event
  per segment boundary and deduplicate by tick.
- A step-limited smoothing pass across consecutive segments keeps the curve
  from jumping between phrases, while still following the audio dynamics.
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


def estimate_loudness_curve(
    audio_path: Path | str,
    *,
    frame_seconds: float = 0.1,
    hop_seconds: float = 0.05,
    smoothing_seconds: float = 0.3,
    db_floor_percentile: float = 10.0,
    db_ceiling_percentile: float = 90.0,
    cc_min: int = 32,
    cc_max: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a CC-style loudness envelope over the whole audio.

    Returns (times_seconds, cc_values) sampled on the hop grid.  The mapping
    is normalized against robust (percentile) loudness bounds of the whole
    stem so that the curve captures *relative* dynamics.
    """
    waveform, source_sr = sf.read(
        str(audio_path), dtype="float32", always_2d=True
    )
    if waveform.shape[1] > 2:
        waveform = waveform[:, :2]
    mono = waveform.mean(axis=1).astype(np.float64)

    frame = max(1, int(round(float(frame_seconds) * float(source_sr))))
    hop = max(1, int(round(float(hop_seconds) * float(source_sr))))
    sample_count = int(mono.shape[0])
    if sample_count <= frame:
        window_starts = np.array([0], dtype=np.int64)
    else:
        window_starts = np.arange(0, sample_count - frame + 1, hop, dtype=np.int64)

    squared = mono * mono
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    window_ends = window_starts + frame
    rms = np.sqrt(
        np.maximum(
            (cumulative[window_ends] - cumulative[window_starts]) / float(frame),
            1e-12,
        )
    )
    db = 20.0 * np.log10(rms + 1e-12)

    kernel = max(1, int(round(float(smoothing_seconds) / max(float(hop_seconds), 1e-6))))
    if len(db) >= kernel:
        padded = np.pad(db, (kernel // 2, kernel - 1 - kernel // 2), mode="edge")
        kernel_array = np.ones(kernel, dtype=np.float64) / float(kernel)
        db = np.convolve(padded, kernel_array, mode="valid")

    if len(db) == 0:
        return np.array([0.0]), np.array([int(cc_min)], dtype=np.int64)

    floor = float(np.percentile(db, db_floor_percentile))
    ceiling = float(np.percentile(db, db_ceiling_percentile))
    span = max(ceiling - floor, 3.0)
    cc_values = np.clip(
        np.round(
            (db - floor) / span * float(cc_max - cc_min) + float(cc_min)
        ),
        cc_min,
        cc_max,
    ).astype(np.int64)
    times = window_starts.astype(np.float64) / float(source_sr)
    return times, cc_values


def segment_notes(
    notes: Sequence[pretty_midi.Note],
    *,
    gap_seconds: float = 0.35,
) -> list[tuple[float, float]]:
    """Group notes into phrase segments by merging near/overlapping onsets.

    Notes whose onsets are separated by more than ``gap_seconds`` (with no
    overlap) start a new segment.  Returns a list of (start, end) intervals.
    """
    if not notes:
        return []
    events = sorted((float(note.start), float(note.end)) for note in notes)
    segments: list[tuple[float, float]] = []
    segment_start, segment_end = events[0]
    for start, end in events[1:]:
        if start <= segment_end + gap_seconds:
            segment_end = max(segment_end, end)
        else:
            segments.append((segment_start, segment_end))
            segment_start, segment_end = start, end
    segments.append((segment_start, segment_end))
    return segments


def merge_segment_cc(
    segments: Sequence[tuple[float, float]],
    curve_times: np.ndarray,
    curve_values: np.ndarray,
    *,
    cc_min: int,
    cc_max: int,
    max_step: int,
) -> list[tuple[float, int]]:
    """Merge per-segment targets into a smooth, conflict-free CC event list.

    The target of each segment is the mean loudness CC over that interval.
    Consecutive segments are constrained by ``max_step`` so the curve cannot
    jump abruptly.  At most one event is produced per segment boundary.
    """
    if not segments:
        return []
    targets: list[float] = []
    for start, end in segments:
        mask = (curve_times >= start) & (curve_times <= end)
        if mask.any():
            targets.append(float(np.mean(curve_values[mask])))
        else:
            index = int(np.searchsorted(curve_times, start, side="right")) - 1
            index = max(0, min(index, int(curve_values.shape[0]) - 1))
            targets.append(float(curve_values[index]))

    events: list[tuple[float, int]] = []
    previous: float | None = None
    for (start, _end), target in zip(segments, targets):
        if previous is None:
            value = float(np.clip(target, cc_min, cc_max))
        else:
            value = float(
                np.clip(previous + (target - previous), previous - max_step, previous + max_step)
            )
        cc_value = int(round(value))
        cc_value = max(cc_min, min(cc_max, cc_value))
        events.append((start, cc_value))
        previous = float(cc_value)
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
    gap_seconds: float = 0.35,
    cc_min: int = 32,
    cc_max: int = 120,
    max_step: int = 12,
) -> int:
    """Write CC events onto sustained instruments of ``midi`` in place.

    Returns the number of instruments that received CC events.  Existing CC
    events with the same controller number are replaced, and events are
    deduplicated by tick so no two CC messages share a tick on one track.
    """
    changed_instruments = 0
    for instrument in midi.instruments:
        if not _is_sustained(int(instrument.program), bool(instrument.is_drum), sustained_ranges):
            continue
        if not instrument.notes:
            continue

        instrument.control_changes = [
            control
            for control in instrument.control_changes
            if int(control.number) != int(cc)
        ]
        segments = segment_notes(instrument.notes, gap_seconds=gap_seconds)
        if not segments:
            continue
        events = merge_segment_cc(
            segments,
            curve_times,
            curve_values,
            cc_min=cc_min,
            cc_max=cc_max,
            max_step=max_step,
        )

        # Emit a state-defining event at time 0, then one per segment boundary.
        by_tick: dict[int, int] = {0: events[0][1]}
        for start, value in events:
            by_tick[int(midi.time_to_tick(start))] = value
        for tick in sorted(by_tick):
            instrument.control_changes.append(
                pretty_midi.ControlChange(
                    number=int(cc),
                    value=int(by_tick[tick]),
                    time=float(midi.tick_to_time(tick)),
                )
            )
        instrument.control_changes.sort(key=lambda control: control.time)
        changed_instruments += 1
    return changed_instruments


def predict_expression_for_stem_midis(
    stem_midis: Mapping[str, Path | str] | Sequence[Path | str],
    stem_audios: Path | str | Mapping[str, Path | str] | None = None,
    *,
    output_dir: Path | str | None = None,
    in_place: bool = False,
    cc: int = 11,
    sustained_ranges: Sequence[tuple[int, int]] = SUSTAINED_PROGRAM_RANGES,
    gap_seconds: float = 0.35,
    frame_seconds: float = 0.1,
    hop_seconds: float = 0.05,
    smoothing_seconds: float = 0.3,
    cc_min: int = 32,
    cc_max: int = 120,
    max_step: int = 12,
    quiet: bool = False,
) -> dict[str, Path]:
    """Apply CC expression curves to each stem MIDI using its full audio.

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

        curve_times, curve_values = estimate_loudness_curve(
            audio_path,
            frame_seconds=frame_seconds,
            hop_seconds=hop_seconds,
            smoothing_seconds=smoothing_seconds,
            cc_min=cc_min,
            cc_max=cc_max,
        )
        midi_obj = pretty_midi.PrettyMIDI(str(midi_path))
        changed = apply_expression_to_midi(
            midi_obj,
            curve_times,
            curve_values,
            cc=cc,
            sustained_ranges=sustained_ranges,
            gap_seconds=gap_seconds,
            cc_min=cc_min,
            cc_max=cc_max,
            max_step=max_step,
        )
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
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=0.35,
        help="Max note gap (s) within one phrase segment.",
    )
    parser.add_argument("--frame-seconds", type=float, default=0.1, help="Loudness frame (s).")
    parser.add_argument("--hop-seconds", type=float, default=0.05, help="Loudness hop (s).")
    parser.add_argument("--smoothing-seconds", type=float, default=0.3, help="Curve smoothing (s).")
    parser.add_argument("--cc-min", type=int, default=32, help="Minimum CC value.")
    parser.add_argument("--cc-max", type=int, default=120, help="Maximum CC value.")
    parser.add_argument(
        "--max-step",
        type=int,
        default=12,
        help="Maximum CC change between consecutive segments.",
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
        gap_seconds=args.gap_seconds,
        frame_seconds=args.frame_seconds,
        hop_seconds=args.hop_seconds,
        smoothing_seconds=args.smoothing_seconds,
        cc_min=args.cc_min,
        cc_max=args.cc_max,
        max_step=args.max_step,
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
