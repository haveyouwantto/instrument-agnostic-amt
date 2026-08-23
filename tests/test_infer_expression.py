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
    # violin (pitch 67 = G4 = 392 Hz) is loud in the first half, quiet in the second.
    violin_amplitude = np.where(time < 2.0, 0.5, 0.05)
    # flute (pitch 71 = B4 = 494 Hz) is quiet in the first half, loud in the second.
    flute_amplitude = np.where(time < 2.0, 0.05, 0.5)
    mono = (
        violin_amplitude * np.sin(2.0 * np.pi * 392.0 * time)
        + flute_amplitude * np.sin(2.0 * np.pi * 494.0 * time)
    ).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    return stereo, sample_rate


def _make_stem_midi() -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(resolution=480)
    violin = pretty_midi.Instrument(program=40, name="violin")  # G4 = pitch 67
    flute = pretty_midi.Instrument(program=73, name="flute")  # B4 = pitch 71
    piano = pretty_midi.Instrument(program=0, name="piano")
    violin.notes = [
        pretty_midi.Note(velocity=90, pitch=67, start=0.0, end=1.0),
        pretty_midi.Note(velocity=90, pitch=67, start=1.0, end=2.0),
        pretty_midi.Note(velocity=90, pitch=67, start=2.0, end=3.0),
        pretty_midi.Note(velocity=90, pitch=67, start=3.0, end=4.0),
    ]
    flute.notes = [
        pretty_midi.Note(velocity=90, pitch=71, start=0.0, end=1.0),
        pretty_midi.Note(velocity=90, pitch=71, start=1.0, end=2.0),
        pretty_midi.Note(velocity=90, pitch=71, start=2.0, end=3.0),
        pretty_midi.Note(velocity=90, pitch=71, start=3.0, end=4.0),
    ]
    piano.notes = [
        pretty_midi.Note(velocity=80, pitch=45, start=0.0, end=0.5),  # A2 = 110 Hz
        pretty_midi.Note(velocity=80, pitch=45, start=1.0, end=1.5),
    ]
    midi.instruments = [violin, flute, piano]
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
    flute = next(instrument for instrument in result.instruments if instrument.program == 73)
    piano = next(instrument for instrument in result.instruments if instrument.program == 0)

    assert piano.control_changes == []

    violin_events = sorted(violin.control_changes, key=lambda event: event.time)
    flute_events = sorted(flute.control_changes, key=lambda event: event.time)
    assert all(event.number == 11 for event in violin_events + flute_events)
    assert max(event.value for event in violin_events) == 127  # normalized peak
    assert max(event.value for event in flute_events) == 127

    times = [event.time for event in violin_events]
    assert all(a < b for a, b in zip(times, times[1:]))
    median_interval = float(np.median(np.diff(times)))
    assert median_interval <= 0.05  # millisecond-level resolution

    # Each instrument follows its own loudness, not the stem's.
    assert _value_at_or_before(violin_events, 1.0) == 127  # violin loud in first half
    assert _value_at_or_before(violin_events, 3.0) < 100  # violin quiet in second half
    assert _value_at_or_before(flute_events, 1.0) < 100  # flute quiet in first half
    assert _value_at_or_before(flute_events, 3.0) == 127  # flute loud in second half


def _value_at_or_before(
    events: list[pretty_midi.ControlChange],
    target_time: float,
) -> int | None:
    """CC state at ``target_time`` (the last event at or before it)."""
    value = None
    for event in events:
        if event.time > target_time:
            break
        value = int(event.value)
    return value


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


def test_frame_events_fade_in_out_keep_transients() -> None:
    midi = pretty_midi.PrettyMIDI(resolution=480)
    notes = [pretty_midi.Note(velocity=90, pitch=67, start=0.5, end=1.5)]
    curve_times = np.arange(0.4, 1.6, 0.02)
    # Alternating quiet/loud frames inside the phrase to expose transients.
    curve_values = np.where(
        np.arange(int(curve_times.shape[0])) % 2 == 0,
        10.0,
        40.0,
    )
    events = build_frame_cc_events(
        midi,
        curve_times,
        curve_values,
        notes,
        interval_seconds=0.02,
        cc_min=8,
        cc_max=127,
        fade_seconds=0.15,
    )
    assert events
    fade_in = [event for event in events if event[0] < 0.5]
    fade_out = [event for event in events if event[0] > 1.5]
    assert fade_in
    assert fade_out
    assert abs(fade_in[0][0] - 0.35) < 0.02  # starts one fade length before
    assert fade_in[0][1] == 8  # ramp starts from the rest value
    fade_in_values = [value for _time, value in fade_in]
    assert fade_in_values == sorted(fade_in_values)  # linear increase before start
    assert fade_out[-1][1] == 8  # ramp ends at the rest value
    fade_out_values = [value for _time, value in fade_out]
    assert fade_out_values == sorted(fade_out_values, reverse=True)  # linear decrease
    # Intra-phrase transients are NOT flattened by a step limit.
    values = [event[1] for event in events]
    assert max(abs(b - a) for a, b in zip(values, values[1:])) > 8


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
