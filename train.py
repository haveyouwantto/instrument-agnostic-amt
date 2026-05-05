import argparse
import logging
import os
import random
from typing import Any
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dataset import StemDataset
from losses import compute_losses
from models.model import AudioSemiCRFTransformer, SemiCRFModelConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """カスタムバッチ作成関数。テンソル以外のデータ(interval_targets)も適切にまとめる。"""
    audio = torch.stack([b["audio"] for b in batch])
    frame_active_targets = torch.stack([b["frame_active_targets"] for b in batch])
    frame_instrument_targets = torch.stack(
        [b["frame_instrument_targets"] for b in batch]
    )
    interval_targets = [b["interval_targets"] for b in batch]
    valid_audio_frames = torch.tensor(
        [b.get("valid_audio_frames", b["audio"].shape[-1]) for b in batch],
        dtype=torch.long,
    )
    # 楽器ラベルなしデータの楽器分類ロスマスク (True = マスクしてロス計算しない)
    mask_instrument_loss = torch.tensor(
        [b.get("mask_instrument_loss", False) for b in batch],
        dtype=torch.bool,
    )
    return {
        "audio": audio,
        "frame_active_targets": frame_active_targets,
        "frame_instrument_targets": frame_instrument_targets,
        "interval_targets": interval_targets,
        "valid_audio_frames": valid_audio_frames,
        "mask_instrument_loss": mask_instrument_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Stem-based AMT Model")
    parser.add_argument("--manifest_path", type=str, default="manifest.csv")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="dataset_config.yaml",
        help="Path to dataset config YAML (dataset_config.yaml). Overrides manifest_path.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of optimizer steps for linear learning-rate warmup",
    )
    parser.add_argument(
        "--epochs", type=int, default=3000, help="Number of training epochs"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Audio sampling rate"
    )
    parser.add_argument(
        "--window_ms", type=int, default=8000, help="Audio window size in milliseconds"
    )
    parser.add_argument(
        "--no_amp", action="store_true", help="Disable mixed precision training"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--project_name",
        type=str,
        default="instrument_agnostic_amt",
        help="Wandb project name",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--ir_folder",
        type=str,
        default=None,
        help="Path to IRs folder for reverb augmentation",
    )
    parser.add_argument(
        "--noise_folder",
        type=str,
        default=None,
        help="Path to noise folder for background noise augmentation",
    )
    parser.add_argument(
        "--drum_folder",
        type=str,
        default=None,
        help="Path to separate drum folder for drum robustness augmentation",
    )
    parser.add_argument(
        "--p_drum_mix",
        type=float,
        default=0.1,
        help="Probability of mixing a separate drum track",
    )
    parser.add_argument(
        "--p_augment",
        type=float,
        default=1.0,
        help="Probability of applying audio augmentations",
    )
    parser.add_argument(
        "--p_intra_drop",
        type=float,
        default=0.3,
        help="Probability of dropping intra-song stems",
    )
    parser.add_argument(
        "--p_cross_mix",
        type=float,
        default=0.5,
        help="Probability of mixing cross-song stems",
    )
    parser.add_argument(
        "--p_use_stems_augments",
        type=float,
        default=0.5,
        help="Probability of loading reverb-processed stems from stems_augments folder",
    )
    parser.add_argument(
        "--sa_freq_max",
        type=int,
        default=0,
        help="SpecAugment maximum frequency mask bins",
    )
    parser.add_argument(
        "--sa_time_max",
        type=int,
        default=0,
        help="SpecAugment maximum time mask frames",
    )
    parser.add_argument(
        "--sa_num_freq",
        type=int,
        default=0,
        help="SpecAugment number of frequency masks",
    )
    parser.add_argument(
        "--sa_num_time", type=int, default=0, help="SpecAugment number of time masks"
    )
    parser.add_argument(
        "--sa_p", type=float, default=0.0, help="SpecAugment probability"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_interval", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to pre-trained checkpoint for weight initialization",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    if args.wandb:
        if not HAS_WANDB:
            logger.warning(
                "wandb is not installed. Please `pip install wandb` to use it. Falling back to console logging."
            )
            args.wandb = False
        else:
            wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Initializing Dataset and DataLoader...")
    dataset = StemDataset(
        manifest_path=args.manifest_path,
        dataset_config_path=args.dataset_config,
        window_ms=args.window_ms,
        sample_rate=args.sample_rate,
        p_intra_drop=args.p_intra_drop,
        p_cross_mix=args.p_cross_mix,
        p_augment=args.p_augment,
        p_use_stems_augments=args.p_use_stems_augments,
        ir_folder=args.ir_folder,
        noise_folder=args.noise_folder,
        drum_folder=args.drum_folder,
        p_drum_mix=args.p_drum_mix,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    logger.info("Initializing Model...")
    spec_augment_params = {
        "freq_mask_max": args.sa_freq_max,
        "time_mask_max": args.sa_time_max,
        "num_freq_masks": args.sa_num_freq,
        "num_time_masks": args.sa_num_time,
        "p": args.sa_p,
    }
    config = SemiCRFModelConfig(
        sample_rate=dataset.sample_rate,
        hop_length=dataset.hop_length,
        n_fft=dataset.n_fft,
        spec_augment_params=spec_augment_params if args.sa_p > 0.0 else None,
    )
    model = AudioSemiCRFTransformer(config).to(device)

    if args.init_from:
        if os.path.exists(args.init_from):
            logger.info(f"Initializing weights from {args.init_from}...")
            checkpoint = torch.load(args.init_from, map_location=device)
            # 'model_state_dict'キーがあればそれを、なければ辞書全体をロード
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            model.load_state_dict(state_dict)
            logger.info("Weights initialized successfully.")
        else:
            logger.error(f"Checkpoint not found at {args.init_from}")
            raise FileNotFoundError(f"Checkpoint not found at {args.init_from}")

    if args.wandb:
        wandb.config.update({"model_config": asdict(config)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be non-negative")

    def lr_lambda(step: int) -> float:
        if args.warmup_steps == 0:
            return 1.0
        return min((step + 1) / args.warmup_steps, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    logger.info(
        f"Starting training on {device} (AMP: {use_amp}, warmup_steps: {args.warmup_steps})"
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        dataset.set_epoch(epoch)  # シードの更新

        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in progress_bar:
            audio = batch["audio"].to(device)
            valid_audio_frames = batch["valid_audio_frames"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # 混合精度学習のコンテキスト
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(audio, valid_audio_frames=valid_audio_frames)
                total_loss, loss_dict = compute_losses(
                    outputs, batch, args=args, model=model
                )

            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_was_skipped = scaler.get_scale() < scale_before_step
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer_step_was_skipped = False

            if not optimizer_step_was_skipped:
                scheduler.step()

            loss_val = total_loss.item()
            epoch_loss += loss_val
            global_step += 1

            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})

            if args.wandb:
                wandb_log_dict = {
                    f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in loss_dict.items()
                }
                wandb_log_dict["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                wandb.log(wandb_log_dict, step=global_step)

        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(
                args.save_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_epoch_loss,
                    "model_config": asdict(config),
                    "config": {
                        "model_config": asdict(config),
                        "args": vars(args),
                    },
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    if args.wandb:
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
