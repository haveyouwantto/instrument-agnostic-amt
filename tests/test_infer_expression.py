from __future__ import annotations

import numpy as np
import pretty_midi
import soundfile as sf

from instrument_agnostic_amt.expression.cli.infer_expression import (
    apply_expression_to_midi,
    build_frame_cc_events,
    estimate_loudness_curve,
    predict_expression_for_stem_midis,
)


def _make_stem_audio(
    *,
    sample_rate: int = 22050,
    duration_seconds: float = 4.0,
) -> tuple[np.ndarray, int]:
    time = np.linspace(
        0.0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False
    )
    amplitude = np.where(time < 2.0, 0.05, 0.5)
    mono = (amplitude * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    return stereo, sample_rate


def _make_stem_midi() -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(resolution=480)
    violin = pretty_midi.Instrument(program=40, name="violin")
    piano = pretty_midi.Instrument(program=0, name="piano")
    violin.notes = [
        pretty_midi.Note(velocity=90, pitch=67, start=0.0, end=1.0),
        pretty_midi.Note(velocity=90, pitch=69, start=1.0, end=2.0),
        pretty_midi.Note(velocity=90, pitch=71, start=2.5, end=3.5),
        pretty_midi.Note(velocity=90, pitch=72, start=3.5, end=4.0),
    ]
    piano.notes = [
        pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5),
        pretty_midi.Note(velocity=80, pitch=62, start=1.0, end=1.5),
    ]
    midi.instruments = [violin, piano]
    return midi


def test_expression_cc_follows_loudness(tmp_path) -> None:
    stereo, sample_rate = _make_stem_audio()
    audio_path = tmp_path / "song_other.wav"
    sf.write(str(audio_path), stereo, sample_rate)

    midi = _make_stem_midi()
    midi_path = tmp_path / "song_other.mid"
    midi.write(str(midi_path))

    outputs = predict_expression_for_stem_midis(
        {"other": midi_path},
        stem_audios={"other": audio_path},
        output_dir=tmp_path / "out",
    )
    assert len(outputs) == 1

    result = pretty_midi.PrettyMIDI(str(outputs["other"]))
    violin = next(instrument for instrument in result.instruments if instrument.program == 40)
    piano = next(instrument for instrument in result.instruments if instrument.program == 0)

    assert piano.control_changes == []
    cc_events = sorted(violin.control_changes, key=lambda event: event.time)
    assert cc_events
    assert all(event.number == 11 for event in cc_events)
    assert max(event.value for event in cc_events) == 127  # normalized peak

    times = [event.time for event in cc_events]
    assert all(a < b for a, b in zip(times, times[1:]))
    median_interval = float(np.median(np.diff(times)))
    assert median_interval <= 0.05  # millisecond-level resolution

    quiet_values = [event.value for event in cc_events if event.time < 2.0]
    loud_values = [event.value for event in cc_events if event.time >= 2.0]
    assert quiet_values
    assert loud_values
    assert float(np.mean(loud_values)) > float(np.mean(quiet_values))


def test_frame_events_are_millisecond_resolution() -> None:
    midi = pretty_midi.PrettyMIDI(resolution=480)
    notes = [
        pretty_midi.Note(velocity=90, pitch=67, start=0.0, end=4.0),
    ]
    curve_times = np.arange(0.0, 4.0, 0.0116)
    curve_values = 20.0 + 20.0 * np.abs(np.sin(curve_times * 2.0))
    events = build_frame_cc_events(
        midi,
        curve_times,
        curve_values,
        notes,
        interval_seconds=0.02,
        cc_min=8,
        cc_max=127,
    )
    assert len(events) >= 100  # ~1 event per 20 ms over 4 s
    times = [event[0] for event in events]
    assert all(a < b for a, b in zip(times, times[1:]))


def test_apply_expression_replaces_existing_cc(tmp_path) -> None:
    audio_path = tmp_path / "audio.wav"
    stereo, sample_rate = _make_stem_audio()
    sf.write(str(audio_path), stereo, sample_rate)
    curve_times, curve_values = estimate_loudness_curve(audio_path)

    midi = _make_stem_midi()
    violin = midi.instruments[0]
    violin.control_changes = [
        pretty_midi.ControlChange(number=11, value=127, time=0.0),
        pretty_midi.ControlChange(number=11, value=30, time=1.0),
    ]
    apply_expression_to_midi(midi, curve_times, curve_values)
    numbers = {event.number for event in violin.control_changes}
    assert numbers == {11}
    assert all(8 <= event.value <= 127 for event in violin.control_changes)
    assert len(violin.control_changes) >= 1

