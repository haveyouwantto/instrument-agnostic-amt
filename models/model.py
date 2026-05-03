from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from instrument_classes import NUM_INSTRUMENT_CLASSES

from .transcription_model import AudioFeatureExtractor, Backbone

from .interval_boundaries import gather_interval_endpoint_features

MIN_MIDI_PITCH = 21
MAX_MIDI_PITCH = 108
NUM_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1


def compute_model_frames(num_audio_frames: int, n_fft: int, hop_length: int) -> int:
    return math.ceil(num_audio_frames / hop_length)


@dataclass(frozen=True)
class SemiCRFModelConfig:
    sample_rate: int
    hop_length: int
    n_fft: int = 2048
    cqt_fmin: float = 27.5
    cqt_n_bins: int = 312
    cqt_bins_per_octave: int = 36
    cqt_filter_scale: float = 0.475
    harmonics: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    harmonic_dropout_p: float = 0.0
    cqt_log_scale: bool = False
    spec_augment_params: dict[str, float | int] | None = None
    hidden_size: int = 384
    base_ch: int = 64
    encoder_num_layers: int = 6
    encoder_num_heads: int = 12
    dropout: float = 0.1
    use_gradient_checkpoint: bool = True
    pitch_query_count: int = 88
    semi_crf_head_dim: int = 256
    semi_crf_length_scaling: str = "none"
    semi_crf_length_penalty: float = 0.0
    use_interval_boundary_head: bool = True
    num_instrument_classes: int = NUM_INSTRUMENT_CLASSES


class TaskFeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class IntervalScorer(nn.Module):
    def __init__(self, input_dim: int, head_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        self.head_dim = int(head_dim)
        self.proj = nn.Linear(input_dim, self.head_dim * 2 + 1)
        self.query_scale = 1.0 / math.sqrt(float(self.head_dim))

    def forward(
        self,
        pitch_query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pitch_query_features.dim() != 4:
            raise ValueError("pitch_query_features must have shape [B, T, P, D]")

        interval_proj = self.proj(pitch_query_features)
        interval_query, interval_key, interval_diag = torch.split(
            interval_proj,
            [self.head_dim, self.head_dim, 1],
            dim=-1,
        )
        return (
            interval_query * self.query_scale,
            interval_key,
            interval_diag.squeeze(-1),
        )


class IntervalBoundaryPredictor(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 4),
        )

    def forward(self, interval_features: torch.Tensor) -> torch.Tensor:
        if interval_features.dim() != 2:
            raise ValueError("interval_features must have shape [N, D]")
        return self.net(interval_features)


class AudioSemiCRFTransformer(nn.Module):
    def __init__(self, config: SemiCRFModelConfig) -> None:
        super().__init__()
        if config.pitch_query_count <= 0:
            raise ValueError("pitch_query_count must be positive")
        if config.semi_crf_head_dim <= 0:
            raise ValueError("semi_crf_head_dim must be positive")
        if config.semi_crf_length_scaling not in {"linear", "sqrt", "none"}:
            raise ValueError(
                "semi_crf_length_scaling must be one of {'linear', 'sqrt', 'none'}"
            )

        self.config = config
        feature_extractor = AudioFeatureExtractor(
            sampling_rate=config.sample_rate,
            hop_length=config.hop_length,
            cqt_fmin=config.cqt_fmin,
            cqt_n_bins=config.cqt_n_bins,
            cqt_bins_per_octave=config.cqt_bins_per_octave,
            cqt_filter_scale=config.cqt_filter_scale,
            harmonics=config.harmonics,
            harmonic_dropout_p=config.harmonic_dropout_p,
            cqt_log_scale=config.cqt_log_scale,
            spec_augment_params=config.spec_augment_params,
        )
        self.backbone = Backbone(
            feature_extractor=feature_extractor,
            hidden_size=config.hidden_size,
            base_ch=config.base_ch,
            output_dim=config.hidden_size,
            num_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            num_pitch_queries=config.pitch_query_count,
            dropout=config.dropout,
            use_gradient_checkpoint=config.use_gradient_checkpoint,
        )
        self.interval_adapter = TaskFeatureAdapter(
            input_dim=self.backbone.query_feature_dim,
            dropout=config.dropout,
        )
        self.instrument_adapter = TaskFeatureAdapter(
            input_dim=self.backbone.query_feature_dim,
            dropout=config.dropout,
        )
        self.interval_scorer = IntervalScorer(
            input_dim=self.backbone.query_feature_dim,
            head_dim=config.semi_crf_head_dim,
        )
        self.interval_boundary_predictor = (
            IntervalBoundaryPredictor(
                input_dim=self.backbone.query_feature_dim,
                dropout=config.dropout,
            )
            if config.use_interval_boundary_head
            else None
        )
        self.instrument_classifier = nn.Linear(
            self.backbone.query_feature_dim, config.num_instrument_classes
        )

    def supports_interval_boundaries(self) -> bool:
        return self.interval_boundary_predictor is not None

    def predict_interval_boundaries(
        self,
        pitch_query_features: torch.Tensor,
        interval_batch: Sequence[Sequence[Sequence[tuple[int, int]]]],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int]]]:
        if self.interval_boundary_predictor is None:
            empty = pitch_query_features.new_zeros((0, 4))
            return empty, []
        interval_features, entries = gather_interval_endpoint_features(
            pitch_query_features,
            interval_batch,
        )
        if not entries:
            return pitch_query_features.new_zeros((0, 4)), []
        return self.interval_boundary_predictor(interval_features), entries

    def _build_frame_valid_mask(
        self,
        *,
        batch_size: int,
        num_frames: int,
        valid_audio_frames: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if valid_audio_frames is None:
            return torch.ones(
                batch_size,
                num_frames,
                dtype=torch.bool,
                device=device,
            )
        if valid_audio_frames.dim() != 1:
            raise ValueError("valid_audio_frames must be a 1D tensor")

        lengths = [
            compute_model_frames(
                int(frame_count),
                self.config.n_fft,
                self.config.hop_length,
            )
            for frame_count in valid_audio_frames.tolist()
        ]
        lengths_tensor = torch.tensor(
            lengths,
            device=device,
            dtype=torch.long,
        )
        positions = torch.arange(
            num_frames,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        return positions < lengths_tensor.unsqueeze(1)

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        valid_audio_frames: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        backbone_output = self.backbone(waveform)
        pitch_query_features = backbone_output.pitch_query_features
        interval_features = self.interval_adapter(pitch_query_features)
        instrument_features = self.instrument_adapter(pitch_query_features)
        interval_query, interval_key, interval_diag = self.interval_scorer(
            interval_features
        )
        frame_valid_mask = self._build_frame_valid_mask(
            batch_size=int(waveform.shape[0]),
            num_frames=int(interval_query.shape[1]),
            valid_audio_frames=valid_audio_frames,
            device=waveform.device,
        )
        instrument_logits = self.instrument_classifier(instrument_features)

        return {
            "band_features": backbone_output.band_features,
            "interval_query": interval_query,
            "interval_key": interval_key,
            "interval_diag": interval_diag,
            "pitch_query_features": pitch_query_features,
            "interval_features": interval_features,
            "instrument_features": instrument_features,
            "instrument_logits": instrument_logits,
            "frame_valid_mask": frame_valid_mask,
        }
