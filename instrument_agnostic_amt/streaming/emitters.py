"""Incremental note emission for the windowed AMT decoders.

The decoders already carry state across windows, so a live UI only needs a rule
for *when* a note stops changing:

* V2 (``WindowNoteStitcher.consume_window``) only ever extends or replaces the
  last note of a track, so every earlier note is frozen the moment a new one is
  appended behind it.
* V1 appends one raw segment per window and merges them in ``finalize``. That
  merge is a left-to-right scan over the segments of a track, so it can be
  replayed incrementally as the windows arrive.

Both emitters keep an accumulator per track, emit a note once no later window
can reach it, and re-emit the note that is still growing with ``final=False`` so
the piano roll can show it stretch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from ..amt.inference.types import PredictedNote
from ..amt.inference.windowed import WindowNoteStitcher


@dataclass(frozen=True)
class NoteEvent:
    """One note as the frontend should draw it right now."""

    id: str
    instrument: str
    instrument_id: int
    pitch: int
    start: float
    end: float
    velocity: int
    final: bool
    stem: str


class NoteEmitter(Protocol):
    def on_window(self, window_start_frame: int) -> list[NoteEvent]: ...

    def finish(self) -> list[NoteEvent]: ...


@dataclass
class _TrackState:
    current: PredictedNote | None = None
    consumed: int = 0
    emitted: int = 0
    last_open: tuple[int, int] | None = None


class _BaseEmitter:
    def __init__(
        self,
        stitcher: WindowNoteStitcher,
        *,
        stem: str,
        sample_rate: int,
        merge_gap_samples: int,
        merge_onset_samples: int,
        instrument_label: Callable[[int], str],
    ) -> None:
        self.stitcher = stitcher
        self.stem = str(stem)
        self.sample_rate = int(sample_rate)
        self.merge_gap_samples = int(merge_gap_samples)
        self.merge_onset_samples = int(merge_onset_samples)
        self.instrument_label = instrument_label
        self._state: dict[object, _TrackState] = {}

    # ── helpers ──────────────────────────────────────────────────────────────
    def _instrument_id_of(self, note: PredictedNote) -> int:
        raise NotImplementedError

    def _event(
        self,
        track: object,
        index: int,
        note: PredictedNote,
        *,
        final: bool,
    ) -> NoteEvent:
        instrument_id = self._instrument_id_of(note)
        return NoteEvent(
            id=f"{self.stem}:{track}:{index}",
            instrument=self.instrument_label(instrument_id),
            instrument_id=int(instrument_id),
            pitch=int(note.pitch),
            start=round(int(note.start_sample) / self.sample_rate, 4),
            end=round(int(note.end_sample) / self.sample_rate, 4),
            velocity=int(note.velocity),
            final=bool(final),
            stem=self.stem,
        )

    def _flush(
        self, track: object, state: _TrackState, note: PredictedNote
    ) -> list[NoteEvent]:
        """Emit an accumulator that nothing can reach any more."""
        state.last_open = None
        if not note.has_offset:
            # finalize() drops segments that never closed at a real offset.
            return []
        event = self._event(track, state.emitted, note, final=True)
        state.emitted += 1
        return [event]

    @staticmethod
    def _open_key(note: PredictedNote) -> tuple[int, int]:
        return (int(note.start_sample), int(note.end_sample))


class V1NoteEmitter(_BaseEmitter):
    """Replays ``WindowNoteStitcher._merge_nearby_notes`` window by window."""

    def _instrument_id_of(self, note: PredictedNote) -> int:
        # V1 assigns class ids after stitching; during decoding every note still
        # carries instrument_id == 0 and the per-window top candidate.
        if note.instrument_candidates:
            return int(note.instrument_candidates[0])
        return 0

    def _absorb(
        self, track: object, state: _TrackState, note: PredictedNote
    ) -> list[NoteEvent]:
        current = state.current
        if current is None:
            state.current = replace(note)
            return []
        # instrument_id is 0 for every V1 note at this point, so the track key is
        # (pitch, slot) exactly as in _merge_nearby_notes.
        same_track = (
            int(note.pitch) == int(current.pitch)
            and int(note.slot_index) == int(current.slot_index)
        )
        if (
            same_track
            and int(note.start_sample)
            <= int(current.end_sample) + self.merge_gap_samples
            and not note.has_onset
        ):
            self.stitcher._merge_note_segments(current, note, overwrite_offset=True)
            return []
        if (
            same_track
            and abs(int(note.start_sample) - int(current.start_sample))
            <= self.merge_onset_samples
        ):
            self.stitcher._merge_note_segments(current, note, overwrite_offset=False)
            return []
        state.current = replace(note)
        return self._flush(track, state, current)

    def on_window(self, window_start_frame: int) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        pending: list[NoteEvent] = []
        for track, notes in self.stitcher.notes_by_pair.items():
            state = self._state.setdefault(track, _TrackState())
            was_consumed = state.consumed
            while state.consumed < len(notes):
                note = notes[state.consumed]
                state.consumed += 1
                events.extend(self._absorb(track, state, note))
            current = state.current
            if current is None:
                continue
            if current.has_onset and not current.has_offset:
                key = self._open_key(current)
                if state.last_open != key or state.consumed != was_consumed:
                    state.last_open = key
                    pending.append(
                        self._event(track, state.emitted, current, final=False)
                    )
        return events + pending

    def finish(self) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        for track, state in self._state.items():
            notes = self.stitcher.notes_by_pair.get(track, [])
            if notes:
                # Mirror finalize(): the last segment is treated as closed.
                notes[-1].has_offset = True
            while state.consumed < len(notes):
                note = notes[state.consumed]
                state.consumed += 1
                events.extend(self._absorb(track, state, note))
            current = state.current
            if current is not None:
                current.has_offset = True
                events.extend(self._flush(track, state, current))
                state.current = None
        return events


class V2NoteEmitter(_BaseEmitter):
    """Emits everything except the tail of each ``consume_window`` track."""

    def _instrument_id_of(self, note: PredictedNote) -> int:
        # Prefer the note's own top class over the (stem level) refined label so
        # the roll colours every instrument separately.
        if note.instrument_candidates:
            return int(note.instrument_candidates[0])
        return int(note.instrument_id)

    def on_window(self, window_start_frame: int) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        for track, notes in self.stitcher.notes_by_pair.items():
            state = self._state.setdefault(track, _TrackState())
            while state.consumed < len(notes) - 1:
                events.extend(self._flush(track, state, notes[state.consumed]))
                state.consumed += 1
            if not notes:
                continue
            tail = notes[-1]
            if tail.has_onset and not tail.has_offset:
                key = self._open_key(tail)
                if state.last_open != key:
                    state.last_open = key
                    events.append(self._event(track, state.emitted, tail, final=False))
        return events

    def finish(self) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        for track, state in self._state.items():
            notes = self.stitcher.notes_by_pair.get(track, [])
            if not notes:
                continue
            notes[-1].has_offset = True
            while state.consumed < len(notes):
                events.extend(self._flush(track, state, notes[state.consumed]))
                state.consumed += 1
        return events


def make_emitter(
    semi_crf_version: str,
    stitcher: WindowNoteStitcher,
    *,
    stem: str,
    sample_rate: int,
    merge_gap_samples: int,
    merge_onset_samples: int,
    instrument_label: Callable[[int], str],
) -> NoteEmitter:
    emitter_cls = V1NoteEmitter if str(semi_crf_version) == "v1" else V2NoteEmitter
    return emitter_cls(
        stitcher,
        stem=stem,
        sample_rate=sample_rate,
        merge_gap_samples=merge_gap_samples,
        merge_onset_samples=merge_onset_samples,
        instrument_label=instrument_label,
    )
