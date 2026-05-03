from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

Interval = tuple[int, int]
FlattenedIntervalEntry = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class PitchIntervalTargets:
    intervals: list[list[Interval]]
    has_onset: list[list[bool]]
    has_offset: list[list[bool]]
    onset_offsets: list[list[float]]
    offset_offsets: list[list[float]]


def count_pitch_intervals(
    targets_or_intervals: PitchIntervalTargets | Sequence[Sequence[Interval]],
) -> int:
    interval_tracks = (
        targets_or_intervals.intervals
        if isinstance(targets_or_intervals, PitchIntervalTargets)
        else targets_or_intervals
    )
    return sum(len(track_intervals) for track_intervals in interval_tracks)


def flatten_interval_entries(
    intervals_batch: Sequence[Sequence[Sequence[Interval]]],
) -> list[FlattenedIntervalEntry]:
    entries: list[FlattenedIntervalEntry] = []
    for batch_index, sample_intervals in enumerate(intervals_batch):
        for pitch_index, track_intervals in enumerate(sample_intervals):
            for interval_index, (begin, end) in enumerate(track_intervals):
                entries.append(
                    (
                        int(batch_index),
                        int(pitch_index),
                        int(interval_index),
                        int(begin),
                        int(end),
                    )
                )
    return entries


def gather_interval_endpoint_features(
    frame_features: torch.Tensor,
    intervals_batch: Sequence[Sequence[Sequence[Interval]]],
) -> tuple[torch.Tensor, list[FlattenedIntervalEntry]]:
    if frame_features.dim() != 4:
        raise ValueError("frame_features must have shape [B, T, P, D]")

    entries = flatten_interval_entries(intervals_batch)
    feature_dim = int(frame_features.shape[-1])
    if not entries:
        return frame_features.new_zeros((0, feature_dim * 3)), []

    device = frame_features.device
    batch_indices = torch.tensor(
        [entry[0] for entry in entries],
        device=device,
        dtype=torch.long,
    )
    pitch_indices = torch.tensor(
        [entry[1] for entry in entries],
        device=device,
        dtype=torch.long,
    )
    begin_indices = torch.tensor(
        [entry[3] for entry in entries],
        device=device,
        dtype=torch.long,
    )
    end_indices = torch.tensor(
        [entry[4] for entry in entries],
        device=device,
        dtype=torch.long,
    )

    begin_features = frame_features[batch_indices, begin_indices, pitch_indices]
    end_features = frame_features[batch_indices, end_indices, pitch_indices]
    stacked_features = torch.cat(
        [
            begin_features,
            end_features,
            begin_features * end_features,
        ],
        dim=-1,
    )
    return stacked_features, entries


def gather_boundary_targets(
    targets_batch: Sequence[PitchIntervalTargets],
    entries: Sequence[FlattenedIntervalEntry],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not entries:
        empty = torch.zeros((0,), device=device, dtype=torch.float32)
        return empty, empty, empty, empty

    has_onset = torch.tensor(
        [
            float(targets_batch[batch_index].has_onset[pitch_index][interval_index])
            for batch_index, pitch_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    has_offset = torch.tensor(
        [
            float(targets_batch[batch_index].has_offset[pitch_index][interval_index])
            for batch_index, pitch_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    onset_offsets = torch.tensor(
        [
            float(targets_batch[batch_index].onset_offsets[pitch_index][interval_index])
            for batch_index, pitch_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    offset_offsets = torch.tensor(
        [
            float(
                targets_batch[batch_index].offset_offsets[pitch_index][interval_index]
            )
            for batch_index, pitch_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    return has_onset, has_offset, onset_offsets, offset_offsets
