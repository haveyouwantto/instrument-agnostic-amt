import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pretty_midi
import miditoolkit

# --- Constants for Chord/Key ---
# Use standard MIDI key names (avoid A#, D#, G# which cause issues in some MIDI writers)
ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
STANDARD_QUALITIES = [
    "5",
    "",
    "-5",
    "m",
    "dim",
    "aug",
    "sus2",
    "sus4",
    "6",
    "7",
    "M7",
    "m6",
    "m7",
    "mM7",
    "7-5",
    "m7-5",
    "aug7",
    "augM7",
    "7sus4",
    "dim7",
    "add9",
    "madd9",
    "69",
    "7(9)",
    "7(13)",
    "7(b9)",
    "7(#9)",
    "7(#9)",
    "7(b13)",
    "7-5(b13)",
    "M7(9)",
    "M7(b9)",
    "M7(13)",
    "M7(#11)",
    "m69",
    "m7(9)",
    "m7(11)",
    "m7(13)",
    "m7(b9)",
    "mM7(9)",
    "mM7(13)",
    "aug7(9)",
    "augM7(#9)",
    "add9(#11)",
    "m7-5(b9)",
    "7(9,13)",
    "7(9,b13)",
    "7(9,#11)",
    "7(b9,#9)",
    "7(b9,13)",
    "7(b9,b13)",
    "7(b9,#11)",
    "7(#9,13)",
    "7(#9,b13)",
    "7(#11,13)",
    "m7(9,11)",
    "m7(9,13)",
    "M7(9,13)",
    "M7(9,#11)",
    "7(9,#11,13)",
    "7(9,#11,b13)",
    "M7(9,#11,13)",
]


def get_chord_name(root_chord_idx: int) -> str:
    num_q = len(STANDARD_QUALITIES)
    if root_chord_idx >= 12 * num_q:
        return "N"
    root_idx = root_chord_idx // num_q
    q_idx = root_chord_idx % num_q
    return f"{ROOT_NAMES[root_idx]}{STANDARD_QUALITIES[q_idx]}"


def get_bass_name(bass_idx: int) -> str:
    return ROOT_NAMES[bass_idx] if bass_idx < 12 else "N"


def get_key_name(key_idx: int) -> str:
    return ROOT_NAMES[key_idx] if key_idx < 12 else "N"


# --- Post-processing Logic ---


def local_maxima_filtering(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))
    peaks = [
        i
        for i in range(1, len(probs) - 1)
        if probs[i] > probs[i - 1] and probs[i] > probs[i + 1] and probs[i] > threshold
    ]
    return np.array(peaks)


def segment_classification(logits: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    T, C = logits.shape
    boundaries = np.concatenate(([0], peaks, [T])).astype(int)
    result = np.zeros(T, dtype=np.int32)
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if start < end:
            result[start:end] = np.argmax(np.mean(logits[start:end], axis=0))
    return result


# --- MIDI Conversion Logic (User provided) ---


def create_tempo_changes(beat_times: np.ndarray, ppq: int) -> list:
    tempo_changes = []
    for beat_index in range(len(beat_times) - 1):
        interval = beat_times[beat_index + 1] - beat_times[beat_index]
        bpm = 60.0 / interval if interval > 0 else 120.0
        # Guard against extremely low BPM which overflows MIDI's 24-bit tempo field (BPM < 3.57)
        bpm = max(20.0, bpm)
        tick_position = beat_index * ppq
        tempo_changes.append(miditoolkit.TempoChange(bpm, tick_position))
    if tempo_changes:
        tempo_changes.append(
            miditoolkit.TempoChange(
                tempo_changes[-1].tempo, (len(beat_times) - 1) * ppq
            )
        )
    return tempo_changes


def infer_time_signatures(
    beat_times: np.ndarray, downbeats: np.ndarray | None, ppq: int
) -> list:
    if downbeats is None or len(downbeats) < 2:
        return [miditoolkit.TimeSignature(4, 4, 0)]
    beat_idx = [np.argmin(np.abs(beat_times - db)) for db in downbeats]
    ts_list = []
    for cur, nxt in zip(beat_idx[:-1], beat_idx[1:]):
        beats_per_bar = max(nxt - cur, 1)
        ts_list.append(miditoolkit.TimeSignature(beats_per_bar, 4, cur * ppq))
    if ts_list:
        ts_list.append(
            miditoolkit.TimeSignature(
                ts_list[-1].numerator, ts_list[-1].denominator, beat_idx[-1] * ppq
            )
        )
    return ts_list


def seconds_to_tick(time_sec: float, beat_times: np.ndarray, ppq: int) -> int:
    idx = np.searchsorted(beat_times, time_sec, side="right") - 1
    idx = np.clip(idx, 0, len(beat_times) - 2)
    local_start, local_end = beat_times[idx], beat_times[idx + 1]
    progress = (
        (time_sec - local_start) / (local_end - local_start)
        if local_end > local_start
        else 0
    )
    # Clip progress to avoid exploding ticks if time_sec is far beyond beat_times
    progress = np.clip(progress, 0, 1.0)
    return max(0, min(16777215, int(idx * ppq + progress * ppq)))


# --- Integrated Metadata Embedder ---


class MidiMetadataEmbedder:
    def __init__(self, sample_rate: int, hop_length: int):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_duration = hop_length / sample_rate

    def embed_all(
        self,
        original_midi: pretty_midi.PrettyMIDI,
        outputs: Dict[str, np.ndarray],
        disable_beat: bool = False,
        disable_chord: bool = False,
    ) -> miditoolkit.MidiFile:
        ppq = original_midi.resolution

        # 1. Beat/Downbeat Detection
        if not disable_beat:
            beat_probs = 1.0 / (
                1.0 + np.exp(-outputs.get("beat_logits", np.array([-1e9])))
            )
            downbeat_probs = 1.0 / (
                1.0 + np.exp(-outputs.get("downbeat_logits", np.array([-1e9])))
            )

            def get_peaks(probs, threshold=0.3):
                return np.array(
                    [
                        i
                        for i in range(1, len(probs) - 1)
                        if probs[i] > probs[i - 1]
                        and probs[i] > probs[i + 1]
                        and probs[i] > threshold
                    ]
                )

            beat_peaks = get_peaks(beat_probs)
            downbeat_peaks = get_peaks(downbeat_probs)

            if len(beat_peaks) < 2:
                # Fallback if no beats detected
                beat_times = np.linspace(
                    0,
                    original_midi.get_end_time(),
                    int(original_midi.get_end_time() * 2) + 1,
                )  # 120BPM fallback
                downbeat_times = None
            else:
                beat_times = beat_peaks * self.frame_duration
                downbeat_times = downbeat_peaks * self.frame_duration
        else:
            # Force 120BPM grid if beat detection is disabled
            end_time = original_midi.get_end_time()
            beat_times = np.linspace(0, end_time, int(end_time * 2) + 1)
            downbeat_times = None

        # Ensure start at 0.0
        if len(beat_times) > 0 and beat_times[0] > 0.01:
            beat_times = np.insert(beat_times, 0, 0.0)
        if (
            downbeat_times is not None
            and len(downbeat_times) > 0
            and downbeat_times[0] > 0.01
        ):
            downbeat_times = np.insert(downbeat_times, 0, 0.0)
        elif downbeat_times is None:
            downbeat_times = np.array([0.0])

        # 2. Create Miditoolkit object
        result = miditoolkit.MidiFile(ticks_per_beat=ppq)
        result.tempo_changes = create_tempo_changes(beat_times, ppq=ppq)
        result.time_signature_changes = infer_time_signatures(
            beat_times, downbeat_times, ppq=ppq
        )

        # 3. Embed KeySignatures (from PrettyMIDI if available, otherwise from outputs)
        if "key_logits" in outputs and not disable_chord:
            kb_logits = outputs["key_boundary_logits"]
            key_logits = outputs["key_logits"]
            k_peaks = local_maxima_filtering(kb_logits, threshold=0.5)
            key_preds = segment_classification(key_logits, k_peaks)
            boundaries = np.concatenate(([0], k_peaks, [len(kb_logits)])).astype(int)
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                time = start * self.frame_duration
                tick = seconds_to_tick(time, beat_times, ppq)
                key_idx = key_preds[start]
                if key_idx < 12:
                    key_name = ROOT_NAMES[key_idx]
                    result.key_signature_changes.append(
                        miditoolkit.KeySignature(key_name, tick)
                    )

        # 4. Embed Chords as Markers
        if "root_chord_logits" in outputs and not disable_chord:
            cb_logits = outputs["chord_boundary_logits"]
            rc_logits = outputs["root_chord_logits"]
            ba_logits = outputs["bass_logits"]
            c_peaks = local_maxima_filtering(cb_logits, threshold=0.5)
            rc_preds = segment_classification(rc_logits, c_peaks)
            ba_preds = segment_classification(ba_logits, c_peaks)
            boundaries = np.concatenate(([0], c_peaks, [len(cb_logits)])).astype(int)
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                time = start * self.frame_duration
                tick = seconds_to_tick(time, beat_times, ppq)
                chord_name = get_chord_name(rc_preds[start])
                bass_name = get_bass_name(ba_preds[start])
                full_chord = (
                    chord_name
                    if (bass_name == "N" or chord_name.startswith(bass_name))
                    else f"{chord_name}/{bass_name}"
                )
                if full_chord != "N":
                    result.markers.append(
                        miditoolkit.Marker(text=f"Chord: {full_chord}", time=tick)
                    )

        # 5. Copy Instruments and convert seconds to ticks
        for inst in original_midi.instruments:
            new_inst = miditoolkit.Instrument(
                program=inst.program, is_drum=inst.is_drum, name=inst.name
            )
            for note in inst.notes:
                start_tick = seconds_to_tick(note.start, beat_times, ppq)
                end_tick = seconds_to_tick(note.end, beat_times, ppq)
                if end_tick <= start_tick:
                    end_tick = start_tick + 1
                new_inst.notes.append(
                    miditoolkit.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=start_tick,
                        end=end_tick,
                    )
                )

            for cc in inst.control_changes:
                tick = seconds_to_tick(cc.time, beat_times, ppq)
                new_inst.control_changes.append(
                    miditoolkit.ControlChange(
                        number=cc.number, value=cc.value, time=tick
                    )
                )

            result.instruments.append(new_inst)

        return result
