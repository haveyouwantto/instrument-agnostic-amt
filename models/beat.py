import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BeatConfig:
    downbeat_pos_weight: float = 20.0
    beat_pos_weight: float = 5.0
    meter_loss_weight: float = 0.05
    loss_tolerance: int = 1

    def __post_init__(self) -> None:
        if self.downbeat_pos_weight <= 0.0:
            raise ValueError("downbeat_pos_weight must be positive")
        if self.beat_pos_weight <= 0.0:
            raise ValueError("beat_pos_weight must be positive")
        if self.meter_loss_weight < 0.0:
            raise ValueError("meter_loss_weight must be non-negative")
        if self.loss_tolerance < 0:
            raise ValueError("loss_tolerance must be non-negative")


def add_beat_training_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Beat training")
    group.add_argument(
        "--beat_dataset_path",
        type=str,
        default="beat_chord_dataset/beat_dataset",
        help="Path to beat dataset root containing audio/ and label/.",
    )
    group.add_argument(
        "--enable_beat_training",
        action="store_true",
        help="Enable auxiliary beat/downbeat/meter training.",
    )
    group.add_argument(
        "--beat_batch_size",
        type=int,
        default=0,
        help="Beat batch size. Uses --batch_size when set to 0.",
    )
    group.add_argument(
        "--beat_num_workers",
        type=int,
        default=-1,
        help="Beat DataLoader workers. Uses --num_workers when set to -1.",
    )
    group.add_argument(
        "--beat_loss_scale",
        type=float,
        default=0.1,
        help="Scale for beat loss to control its impact on the backbone.",
    )
    group.add_argument(
        "--beat_update_interval",
        type=int,
        default=2,
        help="Run beat auxiliary updates every N AMT steps.",
    )
    group.add_argument(
        "--downbeat_pos_weight",
        "--beat_downbeat_pos_weight",
        dest="beat_downbeat_pos_weight",
        type=float,
        default=20.0,
        help="Positive class weight for downbeat BCE loss.",
    )
    group.add_argument(
        "--beat_pos_weight",
        "--beat_beat_pos_weight",
        dest="beat_beat_pos_weight",
        type=float,
        default=5.0,
        help="Positive class weight for beat BCE loss.",
    )
    group.add_argument(
        "--meter_loss_weight",
        "--beat_meter_loss_weight",
        dest="beat_meter_loss_weight",
        type=float,
        default=0.05,
        help="Weight for meter classification loss.",
    )
    group.add_argument(
        "--loss_tolerance",
        "--beat_loss_tolerance",
        dest="beat_loss_tolerance",
        type=int,
        default=1,
        help="Shift tolerance in frames for beat/downbeat BCE loss.",
    )


def beat_config_from_args(args: Any) -> BeatConfig:
    return BeatConfig(
        downbeat_pos_weight=float(getattr(args, "beat_downbeat_pos_weight", 20.0)),
        beat_pos_weight=float(getattr(args, "beat_beat_pos_weight", 5.0)),
        meter_loss_weight=float(getattr(args, "beat_meter_loss_weight", 0.05)),
        loss_tolerance=int(getattr(args, "beat_loss_tolerance", 1)),
    )


def beat_dataset_has_wav_audio(root: str | Path) -> bool:
    audio_dir = Path(root) / "audio"
    return audio_dir.exists() and any(audio_dir.glob("*.wav"))


def _require_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "BeatDataset requires torchaudio to load audio files"
        ) from exc
    return torchaudio


class BeatDataset(Dataset):
    """
    beat_chord_dataset/beat_dataset を読む Dataset。

    1. wav と JSON ラベルを対応付ける。
    2. downbeat と拍子から beat/downbeat/meter の時刻表現を作る。
    3. 学習時にランダム窓を切り出してフレーム単位ターゲットへ変換する。
    """

    def __init__(
        self,
        root: str | Path,
        *,
        window_ms: int,
        sample_rate: int,
        hop_length: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.label_dir = self.root / "label"
        self.window_ms = int(window_ms)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.seed = int(seed)
        self.epoch = 0

        if self.window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if not self.audio_dir.exists() or not self.label_dir.exists():
            raise FileNotFoundError(
                f"Beat dataset must contain audio/ and label/: {self.root}"
            )

        # 1. wav と JSON ラベルを stem 名で対応付ける。
        #    label/song.beat.beats.json に対して audio/song.wav を探す。
        torchaudio = _require_torchaudio()
        label_suffix = ".beat.beats.json"
        audio_by_stem = {
            path.stem: path for path in self.audio_dir.glob("*.wav") if path.is_file()
        }

        raw_items: List[Dict[str, Any]] = []
        meter_keys: set[Tuple[int, int]] = set()
        for label_path in sorted(self.label_dir.glob(f"*{label_suffix}")):
            stem = label_path.name[: -len(label_suffix)]
            audio_path = audio_by_stem.get(stem)
            if audio_path is None:
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            measures: List[Dict[str, float | int]] = []
            for raw_measure in label_data.get("measures", []):
                meter_num = int(raw_measure["time_sig_num"])
                meter_den = int(raw_measure["time_sig_den"])
                if meter_num <= 0 or meter_den <= 0:
                    continue
                measures.append(
                    {
                        "downbeat_sec": float(raw_measure["downbeat_sec"]),
                        "meter_num": meter_num,
                        "meter_den": meter_den,
                        "tempo_bpm": float(raw_measure.get("tempo_bpm", 0.0)),
                    }
                )
                meter_keys.add((meter_num, meter_den))

            measures.sort(key=lambda measure: float(measure["downbeat_sec"]))
            if not measures:
                continue

            info = torchaudio.info(str(audio_path))
            if info.sample_rate <= 0 or info.num_frames <= 0:
                continue

            raw_items.append(
                {
                    "song_name": stem,
                    "audio_path": audio_path,
                    "source_sample_rate": int(info.sample_rate),
                    "duration_sec": float(info.num_frames) / float(info.sample_rate),
                    "measures": measures,
                }
            )

        # 2. データセット全体に出てくる拍子を class index に変換する。
        #    meter は分類問題として扱うので、(4, 4), (6, 8) などを固定順に並べる。
        self.meter_classes: Tuple[Tuple[int, int], ...] = tuple(sorted(meter_keys))
        self.meter_to_index: Dict[Tuple[int, int], int] = {
            meter: index for index, meter in enumerate(self.meter_classes)
        }
        self.num_meter_classes = len(self.meter_classes)
        if self.num_meter_classes == 0 or not raw_items:
            raise ValueError(f"No usable wav beat samples found in {self.root}")

        self.window_frames = int(round(self.window_ms * self.sample_rate / 1000.0))
        self.window_sec = float(self.window_frames) / float(self.sample_rate)
        self.model_frames = math.ceil(self.window_frames / self.hop_length)

        # 3. downbeat だけのラベルから beat 時刻と meter 区間を作る。
        #    同時に BalancedSoftmaxLoss 用の meter 出現回数をフレーム単位で数える。
        items: List[Dict[str, Any]] = []
        meter_counts = torch.zeros(self.num_meter_classes, dtype=torch.float32)
        for raw_item in raw_items:
            beat_times: List[float] = []
            downbeat_times: List[float] = []
            meter_intervals: List[Tuple[float, float, int]] = []
            measures = raw_item["measures"]
            duration_sec = float(raw_item["duration_sec"])
            total_frames = math.ceil(
                int(round(duration_sec * self.sample_rate)) / self.hop_length
            )

            for index, measure in enumerate(measures):
                start_sec = float(measure["downbeat_sec"])
                meter_num = int(measure["meter_num"])
                meter_den = int(measure["meter_den"])
                tempo_bpm = float(measure["tempo_bpm"])

                if index + 1 < len(measures):
                    end_sec = float(measures[index + 1]["downbeat_sec"])
                elif tempo_bpm > 0.0:
                    measure_sec = meter_num * (4.0 / meter_den) * 60.0 / tempo_bpm
                    end_sec = min(start_sec + measure_sec, duration_sec)
                else:
                    end_sec = duration_sec

                if end_sec <= start_sec:
                    continue

                meter_index = self.meter_to_index[(meter_num, meter_den)]
                meter_intervals.append((start_sec, end_sec, meter_index))
                downbeat_times.append(start_sec)

                # ラベルには downbeat しかないので、小節内を拍子の分子で等分して beat を補間する。
                # 6/8 は 2 拍ではなく 6 拍として扱う。
                measure_duration = end_sec - start_sec
                for beat_index in range(meter_num):
                    beat_times.append(
                        start_sec + measure_duration * beat_index / meter_num
                    )

                # BalancedSoftmaxLoss 用に meter の出現回数をフレーム単位で数える。
                start_frame = max(
                    0,
                    math.floor(start_sec * self.sample_rate / self.hop_length),
                )
                end_frame = min(
                    total_frames,
                    math.ceil(end_sec * self.sample_rate / self.hop_length),
                )
                if end_frame > start_frame:
                    meter_counts[meter_index] += float(end_frame - start_frame)

            if not meter_intervals:
                continue

            item = dict(raw_item)
            item.pop("measures")
            item.update(
                {
                    "beat_times": tuple(beat_times),
                    "downbeat_times": tuple(downbeat_times),
                    "meter_intervals": tuple(meter_intervals),
                }
            )
            items.append(item)

        if not items:
            raise ValueError(f"No usable beat labels found in {self.root}")

        self.items = items
        self.meter_class_counts = meter_counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        rng = random.Random(self.seed + self.epoch * len(self.items) + idx)
        torchaudio = _require_torchaudio()

        # 1. 曲全体から学習用の窓を選ぶ。
        #    完全ランダムではなく、なるべくアノテーション区間を含む範囲から選ぶ。
        max_start = max(0.0, float(item["duration_sec"]) - self.window_sec)
        if max_start <= 0.0:
            window_start_sec = 0.0
        else:
            first_label = item["meter_intervals"][0][0]
            last_label = item["meter_intervals"][-1][1]
            min_start = max(0.0, first_label - self.window_sec)
            max_labeled_start = min(max_start, last_label)
            if max_labeled_start > min_start:
                window_start_sec = rng.uniform(min_start, max_labeled_start)
            else:
                window_start_sec = rng.uniform(0.0, max_start)

        # 2. wav を窓単位で読み、モデルの sample_rate と 2ch 入力にそろえる。
        source_sample_rate = int(item["source_sample_rate"])
        source_offset = int(round(window_start_sec * source_sample_rate))
        source_frames = int(math.ceil(self.window_sec * source_sample_rate))
        audio, source_sample_rate = torchaudio.load(
            str(item["audio_path"]),
            frame_offset=source_offset,
            num_frames=source_frames,
        )

        if audio.shape[0] > 2:
            audio = audio[:2]
        elif audio.shape[0] == 1:
            audio = audio.repeat(2, 1)

        if source_sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=source_sample_rate,
                new_freq=self.sample_rate,
            )

        valid_audio_frames = min(int(audio.shape[-1]), self.window_frames)
        if audio.shape[-1] < self.window_frames:
            audio = F.pad(audio, (0, self.window_frames - audio.shape[-1]))
        elif audio.shape[-1] > self.window_frames:
            audio = audio[:, : self.window_frames]
        audio = audio.contiguous()

        # 3. 出力フレーム数に合わせて beat/downbeat/meter ターゲットを初期化する。
        #    meter_targets は未アノテーション区間を -100 にして loss から外す。
        beat_targets = torch.zeros(self.model_frames, dtype=torch.float32)
        downbeat_targets = torch.zeros(self.model_frames, dtype=torch.float32)
        meter_targets = torch.full((self.model_frames,), -100, dtype=torch.long)
        beat_mask = torch.zeros(self.model_frames, dtype=torch.float32)

        window_end_sec = window_start_sec + self.window_sec
        valid_model_frames = math.ceil(valid_audio_frames / self.hop_length)
        valid_model_frames = min(valid_model_frames, self.model_frames)

        # 4. meter 区間をフレームへ展開し、beat/downbeat loss 用の mask も作る。
        for start_sec, end_sec, meter_index in item["meter_intervals"]:
            overlap_start = max(start_sec, window_start_sec)
            overlap_end = min(end_sec, window_end_sec)
            if overlap_end <= overlap_start:
                continue

            start_frame = max(
                0,
                math.floor(
                    (overlap_start - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                ),
            )
            end_frame = min(
                valid_model_frames,
                math.ceil(
                    (overlap_end - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                ),
            )
            if end_frame > start_frame:
                meter_targets[start_frame:end_frame] = int(meter_index)
                beat_mask[start_frame:end_frame] = 1.0

        # 5. beat/downbeat のイベント時刻を最近傍フレームへ立てる。
        for target, times in (
            (beat_targets, item["beat_times"]),
            (downbeat_targets, item["downbeat_times"]),
        ):
            for event_sec in times:
                if event_sec < window_start_sec or event_sec >= window_end_sec:
                    continue
                frame_index = int(
                    round(
                        (event_sec - window_start_sec)
                        * self.sample_rate
                        / self.hop_length
                    )
                )
                if 0 <= frame_index < valid_model_frames:
                    target[frame_index] = 1.0

        # 6. wav が窓長より短かった padding 部分は loss から外す。
        if valid_model_frames < self.model_frames:
            beat_mask[valid_model_frames:] = 0.0
            meter_targets[valid_model_frames:] = -100

        return {
            "audio": audio,
            "valid_audio_frames": valid_audio_frames,
            "song_name": item["song_name"],
            "window_start_sec": window_start_sec,
            "beat_targets": beat_targets,
            "downbeat_targets": downbeat_targets,
            "meter_targets": meter_targets,
            "beat_mask": beat_mask,
        }


def beat_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "audio": torch.stack([item["audio"] for item in batch]),
        "valid_audio_frames": torch.tensor(
            [item["valid_audio_frames"] for item in batch],
            dtype=torch.long,
        ),
        "beat_targets": torch.stack([item["beat_targets"] for item in batch]),
        "downbeat_targets": torch.stack([item["downbeat_targets"] for item in batch]),
        "meter_targets": torch.stack([item["meter_targets"] for item in batch]),
        "beat_mask": torch.stack([item["beat_mask"] for item in batch]),
        "song_name": [item["song_name"] for item in batch],
        "window_start_sec": torch.tensor(
            [item["window_start_sec"] for item in batch],
            dtype=torch.float32,
        ),
    }


class BalancedSoftmaxLoss(nn.Module):
    def __init__(
        self,
        class_counts: Union[List[int], torch.Tensor],
        tau: float = 1.0,
        ignore_index: int = -100,
    ):
        """
        Args:
            class_counts (Union[List[int], torch.Tensor]):
                各クラスの出現回数のリストまたはテンソル。
                事前に Laplace 平滑化（全カウントに+1するなど）を推奨します。
            tau (float, optional): 補正のスケール係数. Defaults to 1.0.
        """
        super().__init__()

        class_counts = torch.as_tensor(class_counts, dtype=torch.float32)

        # log_prior を計算し、バッファとして登録
        # カウントが0のクラスは-infになるのを防ぐため、非常に小さい値にクリップ
        log_prior = torch.log(torch.clamp(class_counts, min=1e-9))

        self.register_buffer("log_prior", log_prior)
        self.tau = tau
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): モデルの出力ロジット (B, T, C)
            labels (torch.Tensor): 正解ラベル (B, T)

        Returns:
            torch.Tensor: 計算された損失値 (スカラー)
        """
        # 形状を合わせる
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))  # (B*T, C)
            labels = labels.reshape(-1)  # (B*T,)

        # meter が未定義のフレームは ignore_index にして、そのまま落とす。
        valid = labels != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        logits = logits[valid]
        labels = labels[valid]

        # ロジット補正: z_k <- z_k + τ * log(n_k)
        adjusted_logits = logits + self.tau * self.log_prior
        loss = F.cross_entropy(adjusted_logits, labels)
        return loss


def masked_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = (predictions - targets).abs()
    if mask is None:
        return diff.mean()

    weighted = diff * mask.to(diff.dtype)
    normalizer = mask.sum().clamp_min(1.0).to(diff.dtype)
    return weighted.sum() / normalizer


# https://github.com/CPJKU/beat_this/blob/main/beat_this/model/loss.py
class ShiftTolerantBCELoss(torch.nn.Module):
    """
    少しずれた beat/downbeat ラベルを許容する BCE loss。
    予測側を max-pooling し、正解フレーム周辺で最も強い予測に勾配を流す。
    """

    def __init__(self, pos_weight: float = 1, tolerance: int = 1):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.tolerance = tolerance

    def spread(self, x: torch.Tensor, factor: int = 1):
        if self.tolerance == 0:
            return x
        return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, x: torch.Tensor, factor: int = 1):
        return x[..., factor * self.tolerance : -factor * self.tolerance or None]

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # 予測側を tolerance 分だけ広げ、端の不確かなフレームを落とす。
        spreaded_preds = self.crop(self.spread(preds))
        cropped_targets = self.crop(targets, factor=2)
        # 正解 beat 周辺の負例は見ない。padding や未アノテーション区間も mask で落とす。
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:
            look_at = look_at * self.crop(mask, factor=2)
        return F.binary_cross_entropy_with_logits(
            spreaded_preds,
            cropped_targets,
            weight=look_at,
            pos_weight=self.pos_weight,
        )


class BeatHead(nn.Module):
    """beat / downbeat / meter をまとめて出すフレーム単位 head。"""

    def __init__(
        self,
        input_dim: int,
        num_meter_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_meter_classes <= 0:
            raise ValueError("num_meter_classes must be positive")
        if hidden_dim is None:
            hidden_dim = input_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_meter_classes = int(num_meter_classes)

        self.shared = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(self.hidden_dim, self.num_meter_classes + 2)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.dim() != 3:
            raise ValueError("features must have shape [B, T, D]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features last dim must be {self.input_dim}, got {features.shape[-1]}"
            )

        logits = self.proj(self.shared(features))
        beat_logits, downbeat_logits, meter_logits = torch.split(
            logits,
            [1, 1, self.num_meter_classes],
            dim=-1,
        )
        downbeat_logits = downbeat_logits.squeeze(-1)
        beat_logits = beat_logits.squeeze(-1) + downbeat_logits

        return {
            "beat_logits": beat_logits,
            "downbeat_logits": downbeat_logits,
            "meter_logits": meter_logits,
        }


class BeatLoss(nn.Module):
    def __init__(
        self,
        config: BeatConfig,
        meter_class_counts: Union[List[int], torch.Tensor],
    ) -> None:
        super().__init__()
        self.config = config
        self.beat_loss = ShiftTolerantBCELoss(
            pos_weight=config.beat_pos_weight,
            tolerance=config.loss_tolerance,
        )
        self.downbeat_loss = ShiftTolerantBCELoss(
            pos_weight=config.downbeat_pos_weight,
            tolerance=config.loss_tolerance,
        )
        self.meter_loss = BalancedSoftmaxLoss(meter_class_counts)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        beat_logits = outputs.get("beat_logits")
        downbeat_logits = outputs.get("downbeat_logits")
        meter_logits = outputs.get("meter_logits")
        if beat_logits is None or downbeat_logits is None or meter_logits is None:
            raise ValueError("Beat training requires beat/downbeat/meter logits")

        beat_targets = batch["beat_targets"].to(beat_logits.device)
        downbeat_targets = batch["downbeat_targets"].to(downbeat_logits.device)
        meter_targets = batch["meter_targets"].to(meter_logits.device)
        beat_mask = batch.get("beat_mask")
        if beat_mask is not None:
            beat_mask = beat_mask.to(beat_logits.device)

        beat_loss = self.beat_loss(
            beat_logits.unsqueeze(1),
            beat_targets.unsqueeze(1),
            None if beat_mask is None else beat_mask.unsqueeze(1),
        )
        downbeat_loss = self.downbeat_loss(
            downbeat_logits.unsqueeze(1),
            downbeat_targets.unsqueeze(1),
            None if beat_mask is None else beat_mask.unsqueeze(1),
        )
        meter_loss = self.meter_loss(meter_logits, meter_targets)

        total_loss = (
            beat_loss
            + downbeat_loss
            + meter_loss * float(self.config.meter_loss_weight)
        )
        return total_loss, {
            "beat_total_loss": total_loss,
            "beat_loss": beat_loss,
            "downbeat_loss": downbeat_loss,
            "meter_loss": meter_loss,
        }


def train_beat_batch(
    *,
    model: nn.Module,
    batch: Dict[str, Any],
    loss_fn: BeatLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    ema_model: Any = None,
    grad_clip_norm: float = 1.0,
    loss_scale: float = 1.0,
) -> Tuple[float, Dict[str, torch.Tensor], bool]:
    total_loss, loss_dict = compute_beat_batch_loss(
        model=model,
        batch=batch,
        loss_fn=loss_fn,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        device=device,
        loss_scale=loss_scale,
    )

    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_was_skipped = scaler.get_scale() < scale_before_step
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        optimizer_step_was_skipped = False

    if not optimizer_step_was_skipped:
        scheduler.step()
        if ema_model is not None:
            ema_model.update(model)

    return float(total_loss.item()), loss_dict, optimizer_step_was_skipped


def compute_beat_batch_loss(
    *,
    model: nn.Module,
    batch: Dict[str, Any],
    loss_fn: BeatLoss,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    loss_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    audio = batch["audio"]
    valid_audio_frames = batch["valid_audio_frames"]

    with torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        outputs = model(
            audio,
            valid_audio_frames=valid_audio_frames,
            include_amt=False,
            include_beat=True,
            include_chord=False,
        )
        total_loss, loss_dict = loss_fn(outputs, batch)
        total_loss = total_loss * loss_scale

    return total_loss, loss_dict
