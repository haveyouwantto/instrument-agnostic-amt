"""Experimental expression inference entrypoint backed by the expression CLI module."""

from __future__ import annotations

from instrument_agnostic_amt.expression.cli.infer_expression import (
    SUSTAINED_PROGRAM_RANGES,
    apply_expression_to_midi,
    estimate_loudness_curve,
    main,
    merge_expression_midis,
    predict_expression_for_stem_midis,
    segment_notes,
)

__all__ = [
    "main",
    "SUSTAINED_PROGRAM_RANGES",
    "estimate_loudness_curve",
    "segment_notes",
    "apply_expression_to_midi",
    "predict_expression_for_stem_midis",
    "merge_expression_midis",
]

if __name__ == "__main__":
    main()

