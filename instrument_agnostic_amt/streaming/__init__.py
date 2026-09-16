"""Streaming (live piano-roll) transcription on top of the windowed AMT decoders."""

from .emitters import NoteEvent, make_emitter
from .pipeline import StreamOptions, run_streaming_transcription

__all__ = [
    "NoteEvent",
    "StreamOptions",
    "make_emitter",
    "run_streaming_transcription",
]
