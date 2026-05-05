from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from models.interval_boundaries import gather_boundary_targets
from models.model import AudioSemiCRFTransformer
from models.semi_crf import compute_pitch_interval_loss


def compute_losses(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, Any],
    args: Any | None = None,
    model: AudioSemiCRFTransformer | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    semi_crf_loss = outputs["interval_query"].sum() * 0.0
    semi_crf_track_count = torch.tensor(
        0,
        device=outputs["interval_query"].device,
        dtype=torch.long,
    )
    semi_crf_interval_count = torch.tensor(
        0,
        device=outputs["interval_query"].device,
        dtype=torch.long,
    )
    semi_crf_loss_weight = (
        1.0 if args is None else float(getattr(args, "semi_crf_loss_weight", 1.0))
    )
    interval_presence_loss_weight = (
        1.0
        if args is None
        else float(getattr(args, "interval_presence_loss_weight", 1.0))
    )
    interval_offset_loss_weight = (
        1.0
        if args is None
        else float(getattr(args, "interval_offset_loss_weight", 1.0))
    )
    instrument_loss_weight = (
        1.0 if args is None else float(getattr(args, "instrument_loss_weight", 1.0))
    )
    interval_boundary_loss = outputs["interval_query"].sum() * 0.0
    interval_presence_loss = outputs["interval_query"].sum() * 0.0
    interval_offset_loss = outputs["interval_query"].sum() * 0.0
    instrument_loss = outputs["interval_query"].sum() * 0.0
    interval_boundary_interval_count = torch.tensor(
        0,
        device=outputs["interval_query"].device,
        dtype=torch.long,
    )

    # Note: frame_valid_mask is extracted from outputs in this new architecture, not batch
    frame_valid_mask = outputs.get("frame_valid_mask")
    interval_query = outputs.get("interval_query")
    interval_key = outputs.get("interval_key")
    interval_diag = outputs.get("interval_diag")
    pitch_query_features = outputs.get("pitch_query_features")
    interval_features = outputs.get("interval_features")
    interval_targets = batch.get("interval_targets")

    if (
        interval_query is None
        or interval_key is None
        or interval_diag is None
        or interval_targets is None
        or frame_valid_mask is None
    ):
        raise ValueError("SemiCRF training requires interval outputs and targets")

    valid_lengths = frame_valid_mask.to(dtype=torch.long).sum(dim=-1)
    length_scaling = "linear"
    length_penalty = 0.0
    if model is not None and hasattr(model, "config"):
        length_scaling = model.config.semi_crf_length_scaling
        length_penalty = model.config.semi_crf_length_penalty
    elif args is not None:
        length_scaling = str(getattr(args, "semi_crf_length_scaling", "linear"))
        length_penalty = float(getattr(args, "semi_crf_length_penalty", 0.0))

    semi_crf_loss_value, track_count, interval_count = compute_pitch_interval_loss(
        interval_query,
        interval_key,
        interval_diag,
        [target.intervals for target in interval_targets],
        valid_lengths,
        length_scaling=length_scaling,
        length_penalty=length_penalty,
        track_batch_size=(
            128
            if args is None
            else int(getattr(args, "semi_crf_track_batch_size", 128))
        ),
    )
    semi_crf_loss = semi_crf_loss_value
    semi_crf_track_count = torch.tensor(
        track_count,
        device=interval_query.device,
        dtype=torch.long,
    )
    semi_crf_interval_count = torch.tensor(
        interval_count,
        device=interval_query.device,
        dtype=torch.long,
    )
    total_loss = semi_crf_loss * semi_crf_loss_weight

    if model is not None and model.supports_interval_boundaries():
        boundary_features = (
            interval_features if interval_features is not None else pitch_query_features
        )
        if boundary_features is None:
            raise ValueError("interval boundary loss requires interval features")
        boundary_logits, entries = model.predict_interval_boundaries(
            boundary_features,
            [target.intervals for target in interval_targets],
        )
        if entries:
            (
                has_onset,
                has_offset,
                onset_offsets,
                offset_offsets,
            ) = gather_boundary_targets(
                interval_targets,
                entries,
                device=boundary_logits.device,
            )
            presence_logits, offset_logits = boundary_logits.chunk(2, dim=-1)
            boundary_targets = torch.stack([has_onset, has_offset], dim=-1)
            interval_presence_loss = F.binary_cross_entropy_with_logits(
                presence_logits,
                boundary_targets,
            )

            offset_targets = torch.stack([onset_offsets, offset_offsets], dim=-1)
            offset_targets = torch.clamp(offset_targets, 0.0, 1.0)
            offset_targets = offset_targets * 0.99 + 0.005

            offset_dist = torch.distributions.ContinuousBernoulli(logits=offset_logits)
            # sum over the two offset dimensions, and average over all boundaries in the batch
            interval_offset_loss = (
                -offset_dist.log_prob(offset_targets).sum(dim=-1).mean()
            )

            interval_boundary_loss = interval_presence_loss + interval_offset_loss
            interval_boundary_interval_count = torch.tensor(
                len(entries),
                device=interval_query.device,
                dtype=torch.long,
            )
            total_loss = total_loss + (
                interval_presence_loss * interval_presence_loss_weight
                + interval_offset_loss * interval_offset_loss_weight
            )

    instrument_logits = outputs.get("instrument_logits")
    instrument_targets = batch.get("frame_instrument_targets")
    frame_active_targets = batch.get("frame_active_targets")

    if (
        instrument_logits is not None
        and instrument_targets is not None
        and frame_active_targets is not None
    ):
        device = instrument_logits.device
        instrument_targets = instrument_targets.to(device)
        frame_active_targets = frame_active_targets.to(device)

        # アクティブなフレーム・ピッチだけを抽出して楽器分類ロスを計算する。
        mask = frame_active_targets > 0.5
        # frame_valid_mask がある場合は、有効な音声長の範囲だけに制限する。
        if frame_valid_mask is not None:
            # frame_valid_mask: [B, T]
            # mask: [B, T, 88]
            valid_mask_expanded = frame_valid_mask.unsqueeze(-1)
            mask = mask & valid_mask_expanded

        # 楽器ラベルがないデータセットのサンプルをマスクから除外する
        mask_instrument_loss_flag = batch.get("mask_instrument_loss")
        if mask_instrument_loss_flag is not None:
            # mask_instrument_loss_flag: [B] (True = ロス計算しない)
            # mask: [B, T, 88] → 対象サンプルの全フレーム・ピッチを False にする
            exclude_mask = mask_instrument_loss_flag.to(device).view(-1, 1, 1)
            mask = mask & ~exclude_mask

        if mask.sum() > 0:
            active_logits = instrument_logits[mask]  # [N, C]
            active_targets = instrument_targets[mask]  # [N, C]

            instrument_loss = F.binary_cross_entropy_with_logits(
                active_logits, active_targets
            )

        total_loss = total_loss + (instrument_loss * instrument_loss_weight)

    return total_loss, {
        "total_loss": total_loss,
        "semi_crf_loss": semi_crf_loss,
        "semi_crf_track_count": semi_crf_track_count,
        "semi_crf_interval_count": semi_crf_interval_count,
        "interval_boundary_loss": interval_boundary_loss,
        "interval_presence_loss": interval_presence_loss,
        "interval_offset_loss": interval_offset_loss,
        "interval_boundary_interval_count": interval_boundary_interval_count,
        "instrument_loss": instrument_loss,
    }
