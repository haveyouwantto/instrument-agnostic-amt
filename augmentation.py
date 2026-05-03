import copy
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class AudioAugmentor:
    """
    オーディオ波形に対するデータ拡張（EQ、ピッチシフト(微小)、リバーブ、ノイズ）を適用するクラス。
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        pitch_shift_semitones: Tuple[float, float] = (
            -0.2,
            0.2,
        ),  # マイクロチューニング
        eq_db_range: Tuple[float, float] = (-3.0, 3.0),
        snr_range: Tuple[float, float] = (3.0, 40.0),
        ir_folder: Optional[str | Path] = None,
        noise_folder: Optional[str | Path] = None,
    ):
        try:
            from audiomentations import (
                AddGaussianSNR,
                Compose,
                SomeOf,
                OneOf,
                PitchShift,
                ApplyImpulseResponse,
                AddBackgroundNoise,
                SevenBandParametricEQ,
                LowShelfFilter,
                HighShelfFilter,
                PeakingFilter,
                HighPassFilter,
                PolarityInversion,
                Reverse,
            )
        except ImportError:
            raise ImportError(
                "audiomentations is not installed. Please run `pip install audiomentations`"
            )

        self.sample_rate = sample_rate

        nyq = sample_rate / 2
        high_shelf_max = min(9000.0, nyq * 0.85)

        # 1. 基本変換 (ピッチ微調整、EQ)
        self.basic_transform = Compose(
            [
                PitchShift(
                    min_semitones=pitch_shift_semitones[0],
                    max_semitones=pitch_shift_semitones[1],
                    p=0.5,
                ),
                SevenBandParametricEQ(
                    min_gain_db=eq_db_range[0], max_gain_db=eq_db_range[1], p=0.5
                ),
                SomeOf(
                    (1, 3),
                    [
                        # 低域の太さ・薄さ
                        LowShelfFilter(
                            min_center_freq=60,
                            max_center_freq=220,
                            min_gain_db=-4.0,
                            max_gain_db=2.5,
                            min_q=0.3,
                            max_q=0.8,
                            p=1.0,
                        ),
                        # 高域の明るさ・暗さ
                        HighShelfFilter(
                            min_center_freq=3500,
                            max_center_freq=high_shelf_max,
                            min_gain_db=-4.0,
                            max_gain_db=3.0,
                            min_q=0.3,
                            max_q=0.8,
                            p=1.0,
                        ),
                        # こもり / 箱鳴り
                        PeakingFilter(
                            min_center_freq=180,
                            max_center_freq=500,
                            min_gain_db=-3.0,
                            max_gain_db=2.0,
                            min_q=0.4,
                            max_q=1.2,
                            p=1.0,
                        ),
                        # 中域の張り出し
                        PeakingFilter(
                            min_center_freq=700,
                            max_center_freq=2500,
                            min_gain_db=-2.5,
                            max_gain_db=2.5,
                            min_q=0.4,
                            max_q=1.0,
                            p=1.0,
                        ),
                        # プレゼンス / 刺さり
                        PeakingFilter(
                            min_center_freq=2500,
                            max_center_freq=7000,
                            min_gain_db=-2.5,
                            max_gain_db=2.0,
                            min_q=0.5,
                            max_q=1.5,
                            p=1.0,
                        ),
                    ],
                    p=0.75,
                ),
                # ミックス・マスタリング後の低域整理に近い軽いHPF
                OneOf(
                    [
                        HighPassFilter(
                            min_cutoff_freq=20,
                            max_cutoff_freq=50,
                            min_rolloff=12,
                            max_rolloff=24,
                            zero_phase=True,
                            p=1.0,
                        ),
                        HighPassFilter(
                            min_cutoff_freq=50,
                            max_cutoff_freq=120,
                            min_rolloff=12,
                            max_rolloff=24,
                            zero_phase=True,
                            p=1.0,
                        ),
                    ],
                    weights=[0.85, 0.15],
                    p=0.6,
                ),
                PolarityInversion(p=0.1),
            ]
        )

        # 2. リバーブ (IRコンボリューション)
        self.reverb = None
        if ir_folder is not None:
            ir_path = Path(ir_folder)
            if ir_path.exists():
                # ApplyImpulseResponseはフォルダパス(ir_path)を受け取る仕様
                self.reverb = ApplyImpulseResponse(
                    ir_path=str(ir_path),
                    p=0.5,
                    lru_cache_size=2000,
                    leave_length_unchanged=True,
                )

        # 3. 背景ノイズ・ホワイトノイズ
        noise_transforms = []
        if noise_folder is not None:
            noise_path = Path(noise_folder)
            if noise_path.exists():
                bg_noise_trans = Compose(
                    [
                        PolarityInversion(p=0.5),
                        Reverse(p=0.5),
                    ]
                )
                noise_transforms.append(
                    AddBackgroundNoise(
                        sounds_path=str(noise_path),
                        min_snr_db=snr_range[0],
                        max_snr_db=snr_range[1],
                        p=0.5,
                        lru_cache_size=256,
                        noise_transform=bg_noise_trans,
                    )
                )

        noise_transforms.append(
            AddGaussianSNR(min_snr_db=snr_range[0], max_snr_db=snr_range[1], p=0.5)
        )
        self.noise_transform = Compose(noise_transforms)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        オーディオテンソルにデータ拡張を適用します。
        Args:
            audio: numpy array の波形。Shapeは [channels, frames] を想定。
        Returns:
            拡張された波形。Shapeは [2, frames] (ステレオ化して返す)。
        """
        audio = copy.deepcopy(audio)

        # audiomentations は [channels, frames] のステレオ入力をサポートしているため、
        # そのまま渡すことで左右チャンネルの位相・ステレオ感を維持したまま処理します。
        x = audio.astype(np.float32)

        # 1. 基本変換
        x = self.basic_transform(x, sample_rate=self.sample_rate)

        # 2. リバーブ (Dry/Wetをランダムにブレンド)
        if self.reverb is not None:
            x_reverb = self.reverb(x, sample_rate=self.sample_rate)
            # DryとWet(リバーブ)の比率をランダムに決定
            alpha = random.random()
            x = alpha * x + (1 - alpha) * x_reverb

        # 3. ノイズ
        x = self.noise_transform(x, sample_rate=self.sample_rate)

        # 4. ステレオ・パンニング操作 (低確率でのみ適用)
        if len(x.shape) == 2 and x.shape[0] == 2:
            # L/Rの反転 (確率10%)
            if random.random() < 0.1:
                x = x[::-1, :].copy()

            # ランダムパンニング (左右の音量バランスの変更、確率20%)
            if random.random() < 0.2:
                # pan_factor が 0.5 なら左が半分、右が1.5倍になる（合計2.0を維持）
                pan_factor = random.uniform(0.5, 1.5)
                x[0, :] *= pan_factor
                x[1, :] *= 2.0 - pan_factor

                # パンニングによって1.0を超えた場合のクリッピング防止
                peak = np.abs(x).max()
                if peak > 1.0:
                    x /= peak

        return x
