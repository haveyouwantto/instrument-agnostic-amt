"""Experimental expression inference entrypoint backed by the expression CLI module."""

from __future__ import annotations

from instrument_agnostic_amt.expression.cli.infer_expression import (
    SUSTAINED_PROGRAM_RANGES,
    apply_expression_to_midi,
    apply_expression_to_merged_midi,
    build_frame_cc_events,
    estimate_loudness_curve,
    main,
    merge_expression_midis,
    predict_expression_for_stem_midis,
)

__all__ = [
    "main",
    "SUSTAINED_PROGRAM_RANGES",
    "estimate_loudness_curve",
    "build_frame_cc_events",
    "apply_expression_to_midi",
    "apply_expression_to_merged_midi",
    "predict_expression_for_stem_midis",
    "merge_expression_midis",
]

if __name__ == "__main__":
    main()
