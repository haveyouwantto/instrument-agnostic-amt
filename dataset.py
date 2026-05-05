import csv
import logging
import random
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from models.interval_boundaries import PitchIntervalTargets
from augmentation import AudioAugmentor
from instrument_classes import NUM_INSTRUMENT_CLASSES

logger = logging.getLogger(__name__)

# モデルが対応するMIDIピッチの範囲
NUM_PITCHES = 88
MIN_MIDI_PITCH = 21
MAX_MIDI_PITCH = 108


def _get_instrument_name(stem_name: str) -> str:
    """ステム名から楽器名を抽出 (末尾の _01 等を除去)"""
    parts = stem_name.split("__")
    inst_part = parts[-1] if len(parts) > 1 else stem_name
    return re.sub(r"_\d+$", "", inst_part)


class WindowNotes:
    """ウィンドウ内のノート情報を保持するデータクラス"""

    def __init__(
        self,
        start_ms: np.ndarray,
        end_ms: np.ndarray,
        pitch: np.ndarray,
        velocity: np.ndarray,
        has_onset: np.ndarray,
        has_offset: np.ndarray,
        instrument: np.ndarray | None = None,
    ):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.pitch = pitch
        self.velocity = velocity
        # ウィンドウ内で発音開始・終了したかどうかのフラグ
        self.has_onset = has_onset
        self.has_offset = has_offset
        if instrument is None:
            self.instrument = np.zeros_like(pitch)
        else:
            self.instrument = instrument

    @classmethod
    def empty(cls) -> "WindowNotes":
        return cls(
            start_ms=np.zeros((0,), dtype=np.int64),
            end_ms=np.zeros((0,), dtype=np.int64),
            pitch=np.zeros((0,), dtype=np.int16),
            velocity=np.zeros((0,), dtype=np.int16),
            has_onset=np.zeros((0,), dtype=np.bool_),
            has_offset=np.zeros((0,), dtype=np.bool_),
            instrument=np.zeros((0,), dtype=np.int16),
        )


def _ms_to_sample_index(ms: np.ndarray, *, sample_rate: int) -> np.ndarray:
    """ミリ秒をサンプルインデックスに変換"""
    return np.rint(
        ms.astype(np.float64, copy=False) * float(sample_rate) / 1000.0
    ).astype(np.int64, copy=False)


def _valid_model_pitch_mask(pitch: np.ndarray) -> np.ndarray:
    """対象とするピッチ範囲(21〜108)に収まっているかのマスクを生成"""
    pitch_i64 = pitch.astype(np.int64, copy=False)
    return (pitch_i64 >= MIN_MIDI_PITCH) & (pitch_i64 <= MAX_MIDI_PITCH)


def _map_model_pitch_array(pitch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MIDIピッチ(21~108)をモデルインデックス(0~87)にマッピング"""
    valid_mask = _valid_model_pitch_mask(pitch)
    mapped_pitch = pitch.astype(np.int64, copy=False)[valid_mask] - MIN_MIDI_PITCH
    return mapped_pitch.astype(np.int64, copy=False), valid_mask


def split_window_notes(
    *,
    start_ms: np.ndarray,
    end_ms: np.ndarray,
    pitch: np.ndarray,
    velocity: np.ndarray,
    instrument: np.ndarray,
    window_start_ms: int,
    window_end_ms: int,
    clip_note_end_to_window: bool = True,
) -> tuple[WindowNotes, WindowNotes]:
    """
    指定された時間ウィンドウに含まれるノートを抽出し、
    ウィンドウ開始前から鳴り続けているノート(carry_in)と、
    ウィンドウ内で新しく発音されたノート(body)に分割する。
    """
    max_window_length_ms = int(window_end_ms) - int(window_start_ms)

    def select(note_mask: np.ndarray, *, start_at_zero: bool) -> WindowNotes:
        if not np.any(note_mask):
            return WindowNotes.empty()

        # ウィンドウ開始位置を基準とした相対時間に変換
        rel_start = (
            np.zeros(int(note_mask.sum()), dtype=np.int64)
            if start_at_zero
            else start_ms[note_mask] - window_start_ms
        )
        rel_end = end_ms[note_mask] - window_start_ms

        if clip_note_end_to_window:
            rel_end = np.minimum(rel_end, max_window_length_ms)
        # 最低1msの長さを保証
        rel_end = np.maximum(rel_end, rel_start + 1)

        # ウィンドウ境界を跨いで音が伸びているか判定
        tie_to_next_window = (end_ms[note_mask] > window_end_ms).astype(
            np.bool_, copy=False
        )

        return WindowNotes(
            start_ms=rel_start.astype(np.int64, copy=False),
            end_ms=rel_end.astype(np.int64, copy=False),
            pitch=pitch[note_mask].astype(np.int16, copy=False),
            velocity=velocity[note_mask].astype(np.int16, copy=False),
            instrument=instrument[note_mask].astype(np.int16, copy=False),
            has_onset=np.full(
                int(note_mask.sum()), fill_value=(not start_at_zero), dtype=np.bool_
            ),
            has_offset=np.logical_not(tie_to_next_window).astype(np.bool_, copy=False),
        )

    # carry_in: ウィンドウ開始前がオンセットのノート
    carry_in_mask = (start_ms < window_start_ms) & (end_ms > window_start_ms)
    # body: ウィンドウ内でオンセットがあるノート
    body_mask = (start_ms >= window_start_ms) & (start_ms < window_end_ms)

    return select(carry_in_mask, start_at_zero=True), select(
        body_mask, start_at_zero=False
    )


def concat_window_notes(*note_groups: WindowNotes) -> WindowNotes:
    """複数のWindowNotesオブジェクトを結合する"""
    non_empty_groups = [group for group in note_groups if group.start_ms.size > 0]
    if not non_empty_groups:
        return WindowNotes.empty()

    return WindowNotes(
        start_ms=np.concatenate([group.start_ms for group in non_empty_groups]).astype(
            np.int64, copy=False
        ),
        end_ms=np.concatenate([group.end_ms for group in non_empty_groups]).astype(
            np.int64, copy=False
        ),
        pitch=np.concatenate([group.pitch for group in non_empty_groups]).astype(
            np.int16, copy=False
        ),
        velocity=np.concatenate([group.velocity for group in non_empty_groups]).astype(
            np.int16, copy=False
        ),
        instrument=np.concatenate(
            [group.instrument for group in non_empty_groups]
        ).astype(np.int16, copy=False),
        has_onset=np.concatenate(
            [group.has_onset for group in non_empty_groups]
        ).astype(np.bool_, copy=False),
        has_offset=np.concatenate(
            [group.has_offset for group in non_empty_groups]
        ).astype(np.bool_, copy=False),
    )


def build_frame_note_targets(
    *,
    active_start_ms: np.ndarray,
    active_end_ms: np.ndarray,
    active_pitch: np.ndarray,
    sample_rate: int,
    hop_length: int,
    num_frames: int,
) -> torch.Tensor:
    """フレーム単位のノートアクティベーション（[num_frames, 88]）を生成する"""
    active_targets = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
    if num_frames <= 0 or active_start_ms.size == 0:
        return torch.from_numpy(active_targets)

    start_samples = _ms_to_sample_index(active_start_ms, sample_rate=sample_rate)
    end_samples = _ms_to_sample_index(active_end_ms, sample_rate=sample_rate)

    start_frames = np.clip(start_samples // int(hop_length), 0, num_frames - 1)
    end_frames = (np.maximum(end_samples - 1, 0) // int(hop_length)) + 1
    end_frames = np.clip(np.maximum(end_frames, start_frames + 1), 0, num_frames)

    active_pitches, valid_pitch_mask = _map_model_pitch_array(active_pitch)
    if not np.any(valid_pitch_mask):
        return torch.from_numpy(active_targets)

    start_frames = start_frames[valid_pitch_mask]
    end_frames = end_frames[valid_pitch_mask]

    for start_frame, end_frame, pitch_value in zip(
        start_frames.tolist(), end_frames.tolist(), active_pitches.tolist()
    ):
        if start_frame >= num_frames:
            continue
        active_targets[start_frame:end_frame, pitch_value] = 1.0

    return torch.from_numpy(active_targets)


def build_instrument_targets(
    *,
    active_start_ms: np.ndarray,
    active_end_ms: np.ndarray,
    active_pitch: np.ndarray,
    active_instrument: np.ndarray,
    sample_rate: int,
    hop_length: int,
    num_frames: int,
) -> torch.Tensor:
    """フレーム・ピッチ単位の楽器ラベル（[num_frames, 88, 33]）を生成する"""
    active_targets = np.zeros(
        (num_frames, NUM_PITCHES, NUM_INSTRUMENT_CLASSES), dtype=np.float32
    )
    if num_frames <= 0 or active_start_ms.size == 0 or NUM_INSTRUMENT_CLASSES == 0:
        return torch.from_numpy(active_targets)

    start_samples = _ms_to_sample_index(active_start_ms, sample_rate=sample_rate)
    end_samples = _ms_to_sample_index(active_end_ms, sample_rate=sample_rate)

    start_frames = np.clip(start_samples // int(hop_length), 0, num_frames - 1)
    end_frames = (np.maximum(end_samples - 1, 0) // int(hop_length)) + 1
    end_frames = np.clip(np.maximum(end_frames, start_frames + 1), 0, num_frames)

    active_pitches, valid_pitch_mask = _map_model_pitch_array(active_pitch)
    if not np.any(valid_pitch_mask):
        return torch.from_numpy(active_targets)

    start_frames = start_frames[valid_pitch_mask]
    end_frames = end_frames[valid_pitch_mask]
    instruments = active_instrument[valid_pitch_mask]

    for start_frame, end_frame, pitch_value, inst_id in zip(
        start_frames.tolist(),
        end_frames.tolist(),
        active_pitches.tolist(),
        instruments.tolist(),
    ):
        if start_frame >= num_frames:
            continue
        # マルチホットターゲット
        if 0 <= inst_id < NUM_INSTRUMENT_CLASSES:
            active_targets[start_frame:end_frame, pitch_value, inst_id] = 1.0

    return torch.from_numpy(active_targets)


def build_pitch_interval_targets(
    *,
    active_start_ms: np.ndarray,
    active_end_ms: np.ndarray,
    active_pitch: np.ndarray,
    active_has_onset: np.ndarray,
    active_has_offset: np.ndarray,
    sample_rate: int,
    hop_length: int,
    num_frames: int,
) -> PitchIntervalTargets:
    """Semi-CRFモデル用の詳細なインターバルターゲットを生成する"""
    pitch_intervals: list[list[tuple[int, int]]] = [[] for _ in range(NUM_PITCHES)]
    has_onset_tracks: list[list[bool]] = [[] for _ in range(NUM_PITCHES)]
    has_offset_tracks: list[list[bool]] = [[] for _ in range(NUM_PITCHES)]
    onset_offsets_tracks: list[list[float]] = [[] for _ in range(NUM_PITCHES)]
    offset_offsets_tracks: list[list[float]] = [[] for _ in range(NUM_PITCHES)]

    if num_frames <= 0 or active_start_ms.size == 0:
        return PitchIntervalTargets(
            intervals=pitch_intervals,
            has_onset=has_onset_tracks,
            has_offset=has_offset_tracks,
            onset_offsets=onset_offsets_tracks,
            offset_offsets=offset_offsets_tracks,
        )

    # 続く複雑な処理は、フレーム境界を正確に計算し、複数のノートが重なった場合に
    # 単一の連続したインターバルにマージするためのロジックです。
    start_samples = _ms_to_sample_index(active_start_ms, sample_rate=sample_rate)
    end_samples = _ms_to_sample_index(active_end_ms, sample_rate=sample_rate)

    real_start_frames = start_samples.astype(np.float64, copy=False) / float(hop_length)
    real_end_frames = end_samples.astype(np.float64, copy=False) / float(hop_length)

    raw_start_frames = start_samples // int(hop_length)
    raw_end_frames_exclusive = (np.maximum(end_samples - 1, 0) // int(hop_length)) + 1
    raw_end_frames_inclusive = raw_end_frames_exclusive - 1

    start_frames = np.clip(raw_start_frames, 0, num_frames - 1)
    end_frames_exclusive = np.clip(
        np.maximum(raw_end_frames_exclusive, start_frames + 1), 0, num_frames
    )

    # オフセットの端数計算 (ロス計算時の微細なタイミング補正用)
    onset_offsets = real_start_frames - raw_start_frames
    offset_offsets = real_end_frames - raw_end_frames_inclusive
    mapped_pitch, valid_pitch_mask = _map_model_pitch_array(active_pitch)

    if not np.any(valid_pitch_mask):
        return PitchIntervalTargets(
            intervals=pitch_intervals,
            has_onset=has_onset_tracks,
            has_offset=has_offset_tracks,
            onset_offsets=onset_offsets_tracks,
            offset_offsets=offset_offsets_tracks,
        )

    start_frames = start_frames[valid_pitch_mask]
    end_frames_exclusive = end_frames_exclusive[valid_pitch_mask]
    active_has_onset = active_has_onset[valid_pitch_mask]
    active_has_offset = active_has_offset[valid_pitch_mask]
    onset_offsets = onset_offsets[valid_pitch_mask]
    offset_offsets = offset_offsets[valid_pitch_mask]

    raw_by_pitch: list[list[tuple[int, int, bool, bool, float, float]]] = [
        [] for _ in range(NUM_PITCHES)
    ]
    for (
        start_frame,
        end_frame_exclusive,
        pitch_value,
        has_onset,
        has_offset,
        onset_off,
        offset_off,
    ) in zip(
        start_frames.tolist(),
        end_frames_exclusive.tolist(),
        mapped_pitch.tolist(),
        active_has_onset.tolist(),
        active_has_offset.tolist(),
        onset_offsets.tolist(),
        offset_offsets.tolist(),
    ):
        if start_frame >= num_frames or end_frame_exclusive <= start_frame:
            continue
        raw_by_pitch[pitch_value].append(
            (
                int(start_frame),
                int(end_frame_exclusive - 1),
                bool(has_onset),
                bool(has_offset),
                float(onset_off),
                float(offset_off),
            )
        )

    # ピッチごとにインターバルをソートし、重なりをマージする
    for pitch_value, intervals in enumerate(raw_by_pitch):
        if not intervals:
            continue
        intervals.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        sanitized: list[list[int | bool | float]] = []
        for begin, end, has_onset, has_offset, onset_off, offset_off in intervals:
            # 重複がある場合、前のインターバルを切り詰めるかマージする
            if sanitized and begin <= sanitized[-1][1]:
                prev_begin = int(sanitized[-1][0])
                if begin > prev_begin:
                    sanitized[-1][1] = begin - 1
                    sanitized[-1][3] = True
                    sanitized[-1][5] = 0.5
                else:
                    sanitized.pop()
            if sanitized and begin <= sanitized[-1][1]:
                begin = sanitized[-1][1] + 1
                onset_off = 0.5
            if begin > end:
                continue
            sanitized.append(
                [
                    begin,
                    end,
                    bool(has_onset),
                    bool(has_offset),
                    float(onset_off),
                    float(offset_off),
                ]
            )

        for begin, end, has_onset, has_offset, onset_off, offset_off in sanitized:
            if int(begin) > int(end):
                continue
            pitch_intervals[pitch_value].append((int(begin), int(end)))
            has_onset_tracks[pitch_value].append(bool(has_onset))
            has_offset_tracks[pitch_value].append(bool(has_offset))
            onset_offsets_tracks[pitch_value].append(float(onset_off))
            offset_offsets_tracks[pitch_value].append(float(offset_off))

    return PitchIntervalTargets(
        intervals=pitch_intervals,
        has_onset=has_onset_tracks,
        has_offset=has_offset_tracks,
        onset_offsets=onset_offsets_tracks,
        offset_offsets=offset_offsets_tracks,
    )


def load_audio_window(
    audio_path: str, *, sample_rate: int, window_start_ms: int, window_ms: int
) -> np.ndarray:
    """指定された時間範囲のオーディオを読み込み、モノラル(2ch同じ値)として返す"""
    start_frame = int(round(window_start_ms * sample_rate / 1000.0))
    window_frames = int(round(window_ms * sample_rate / 1000.0))
    audio, _ = sf.read(
        audio_path,
        start=start_frame,
        frames=window_frames,
        dtype="float32",
        always_2d=True,
    )
    # ステレオ対応・モノラル複製
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    audio = audio.transpose(1, 0)  # [channels, frames] に変換

    # 尺が足りない場合はゼロ埋め
    if audio.shape[1] < window_frames:
        padded = np.zeros((audio.shape[0], window_frames), dtype=np.float32)
        padded[:, : audio.shape[1]] = audio
        audio = padded
    return audio.astype(np.float32, copy=False)


def compute_model_frames(audio_frames: int, n_fft: int, hop_length: int) -> int:
    """オーディオのフレーム数から、モデル入力のフレーム数（特徴量サイズ）を計算"""
    return math.ceil(audio_frames / hop_length)


class StemDataset(Dataset):
    """
    ステムオーディオとMIDIペアを読み込み、Intra/Crossオーグメンテーションを適用するデータセット。
    dataset_config_path (YAML) で複数データセットの重みベース混合に対応する。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        dataset_config_path: str | Path | None = None,
        window_ms: int = 5000,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 22050,
        p_intra_drop: float = 0.2,
        p_cross_mix: float = 0.1,
        p_cross_mix_decay: float = 0.3,
        max_cross_stems: int = 5,
        p_augment: float = 0.5,
        p_use_stems_augments: float = 0.0,
        ir_folder: str | Path | None = None,
        noise_folder: str | Path | None = None,
        drum_folder: str | Path | None = None,
        p_drum_mix: float = 0.1,
        seed: int = 42,
    ):
        self.window_ms = int(window_ms)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        # 同一曲からのステムを落とす確率
        self.p_intra_drop = float(p_intra_drop)
        # 別の曲からのステムを混ぜる確率
        self.p_cross_mix = float(p_cross_mix)
        # 別の曲のステムを連続して混ぜる際の減衰係数
        self.p_cross_mix_decay = float(p_cross_mix_decay)
        self.max_cross_stems = int(max_cross_stems)
        self.p_augment = float(p_augment)
        self.p_use_stems_augments = float(p_use_stems_augments)
        self.seed = int(seed)
        self.epoch = 0

        if self.p_augment > 0.0:
            self.augmentor = AudioAugmentor(
                sample_rate=self.sample_rate,
                ir_folder=ir_folder,
                noise_folder=noise_folder,
            )
        else:
            self.augmentor = None

        self.p_drum_mix = float(p_drum_mix)
        self.drum_files: list[str] = []
        if drum_folder is not None and Path(drum_folder).exists():
            for p in Path(drum_folder).rglob("*"):
                if p.is_file() and p.suffix.lower() in [".wav", ".flac", ".mp3"]:
                    self.drum_files.append(str(p))
            if self.drum_files:
                logger.info(f"Found {len(self.drum_files)} drum files in {drum_folder}")
            else:
                logger.warning(f"No audio files found in drum_folder: {drum_folder}")

        self.window_frames = int(round(self.window_ms * self.sample_rate / 1000.0))
        self.model_frames = max(
            0, compute_model_frames(self.window_frames, self.n_fft, self.hop_length)
        )

        self.stems_by_song = defaultdict(list)
        self.all_stems = []

        # データセットグループ: [{name, song_names, weight}, ...]
        self.dataset_groups: list[dict] = []

        if dataset_config_path is not None and Path(dataset_config_path).exists():
            self._load_config(dataset_config_path)
        else:
            # コンフィグなしの場合は単一マニフェストのみ
            self._load_manifest(manifest_path)
            primary_songs = list(self.stems_by_song.keys())
            self.dataset_groups.append(
                {
                    "name": "main",
                    "song_names": primary_songs,
                    "weight": 1.0,
                    "use_for_cross_aug": True,
                }
            )

        # primaryデータセット（最初のグループ）の曲名リスト
        self.primary_song_names = self.dataset_groups[0]["song_names"]

        # 重みから累積確率を計算
        total_weight = sum(group["weight"] for group in self.dataset_groups)
        self._cumulative_probs: list[float] = []
        cumulative = 0.0
        for group in self.dataset_groups:
            cumulative += group["weight"] / total_weight
            self._cumulative_probs.append(cumulative)

        for group in self.dataset_groups:
            probability = group["weight"] / total_weight * 100
            logger.info(
                f"Dataset '{group['name']}': {len(group['song_names'])} songs, "
                f"weight={group['weight']}, prob={probability:.1f}%, cross_aug={group.get('use_for_cross_aug', True)}"
            )

        # Cross augmentation用のグループと累積確率を計算
        self.cross_dataset_groups = [
            g for g in self.dataset_groups if g.get("use_for_cross_aug", True)
        ]
        self._cross_cumulative_probs = []
        if self.cross_dataset_groups:
            total_cross_weight = sum(g["weight"] for g in self.cross_dataset_groups)
            cumulative_cross = 0.0
            for g in self.cross_dataset_groups:
                cumulative_cross += g["weight"] / total_cross_weight
                self._cross_cumulative_probs.append(cumulative_cross)

    def _load_config(self, config_path: str | Path):
        """YAMLコンフィグを読み込み、各データセットのマニフェストをロードする"""
        import yaml

        config_path = Path(config_path)
        config_dir = config_path.parent

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for dataset_entry in config.get("datasets", []):
            manifest_rel = dataset_entry["manifest"]
            manifest_full = config_dir / manifest_rel
            if not manifest_full.exists():
                logger.warning(f"Manifest not found, skipping: {manifest_full}")
                continue

            # ロード前の曲名を記録（新規追加分を特定するため）
            dataset_name = dataset_entry.get("name", manifest_rel)
            existing_songs = set(self.stems_by_song.keys())
            self._load_manifest(manifest_full, song_name_prefix=dataset_name)
            new_songs = [
                name for name in self.stems_by_song if name not in existing_songs
            ]

            self.dataset_groups.append(
                {
                    "name": dataset_entry.get("name", manifest_rel),
                    "song_names": new_songs,
                    "weight": float(dataset_entry.get("weight", 1.0)),
                    "use_for_cross_aug": bool(
                        dataset_entry.get("use_for_cross_aug", True)
                    ),
                    "mask_instrument_loss": bool(
                        dataset_entry.get("mask_instrument_loss", False)
                    ),
                }
            )

    def _load_manifest(self, manifest_path: str | Path, song_name_prefix: str = ""):
        """マニフェストCSVを読み込み、stems_by_songとall_stemsに追加する"""
        manifest_path = Path(manifest_path)
        manifest_dir = manifest_path.parent
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV内のパスはマニフェストファイルからの相対パスなので解決する
                wav_path = str(manifest_dir / row["wav_path"]).replace("\\", "/")
                npz_path = str(manifest_dir / row["npz_path"]).replace("\\", "/")
                # プレフィクス付きの曲名でデータセット間の名前衝突を防ぐ
                song_name = row["song_name"]
                if song_name_prefix:
                    song_name = f"{song_name_prefix}/{song_name}"
                stem_info = {
                    "song_name": song_name,
                    "stem_name": row["stem_name"],
                    "wav_path": wav_path,
                    "npz_path": npz_path,
                    "duration_ms": int(row["duration_ms"]),
                    "end_note_ms": int(row["end_note_ms"]),
                    "note_count": int(row["note_count"]),
                }
                self.stems_by_song[song_name].append(stem_info)
                self.all_stems.append(stem_info)

    def set_epoch(self, epoch: int):
        """学習時のランダムシード制御用"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.primary_song_names)

    def _select_dataset_group(self, rng: random.Random) -> dict:
        """重みに基づいてデータセットグループを選択する"""
        roll = rng.random()
        for group, cumulative_prob in zip(self.dataset_groups, self._cumulative_probs):
            if roll < cumulative_prob:
                return group
        return self.dataset_groups[-1]

    def _select_cross_dataset_group(self, rng: random.Random) -> dict | None:
        """Cross augmentation用の重みに基づいてデータセットグループを選択する"""
        if not self.cross_dataset_groups:
            return None
        roll = rng.random()
        for group, cumulative_prob in zip(
            self.cross_dataset_groups, self._cross_cumulative_probs
        ):
            if roll < cumulative_prob:
                return group
        return self.cross_dataset_groups[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * len(self.primary_song_names) + idx)

        # 重みに基づいてデータセットを選択
        selected_group = self._select_dataset_group(rng)
        if selected_group is self.dataset_groups[0]:
            # primaryデータセット: idxで曲を指定（全曲均等カバー）
            song_name = self.primary_song_names[idx]
        else:
            # extraデータセット: ランダムに曲を選択
            song_name = rng.choice(selected_group["song_names"])

        base_stems = self.stems_by_song[song_name]

        # 1. Intra-drop augmentation: 曲を構成する各ステムを確率で落とす
        selected_base_stems = []
        for stem in base_stems:
            if rng.random() >= self.p_intra_drop:
                selected_base_stems.append(stem)

        # もし全部ドロップしてしまったら、最低1つは残す
        if not selected_base_stems:
            selected_base_stems.append(rng.choice(base_stems))

        # 元曲で残っている楽器の集合を作成
        base_instruments = {
            _get_instrument_name(stem["stem_name"]) for stem in selected_base_stems
        }

        # 2. ウィンドウ開始位置の決定: 選ばれたステムの最大のmin(duration_ms, end_note_ms)を基準にする
        max_effective_end_ms = max(
            min(int(stem["duration_ms"]), int(stem["end_note_ms"]))
            for stem in selected_base_stems
        )
        max_start_ms = max(0, max_effective_end_ms - self.window_ms)
        window_start_ms = rng.randint(0, max_start_ms) if max_start_ms > 0 else 0

        active_stems_with_offset = [
            (stem, window_start_ms) for stem in selected_base_stems
        ]

        # 3. Cross-song mix augmentation: 全く別の曲からステムを混ぜる
        if (
            rng.random() < self.p_cross_mix
            and len(self.all_stems) > 0
            and self.cross_dataset_groups
        ):
            for j in range(self.max_cross_stems):
                # j回目の追加を行うかの継続確率 (j=0は1.0)
                continue_prob = math.exp(-self.p_cross_mix_decay * j)
                if rng.random() >= continue_prob:
                    break

                max_retry = 10
                for _ in range(max_retry):
                    # データセットごとの重みに基づいてグループを選択
                    cross_group = self._select_cross_dataset_group(rng)
                    if cross_group is None:
                        break
                    cross_song_name = rng.choice(cross_group["song_names"])
                    extra_stem = rng.choice(self.stems_by_song[cross_song_name])

                    if extra_stem["song_name"] != song_name:
                        extra_inst = _get_instrument_name(extra_stem["stem_name"])
                        # 同じ楽器は追加しない
                        if extra_inst not in base_instruments:
                            effective_end_ms = min(
                                int(extra_stem["duration_ms"]),
                                int(extra_stem["end_note_ms"]),
                            )
                            stem_max_start = max(0, effective_end_ms - self.window_ms)
                            stem_window_start_ms = (
                                rng.randint(0, stem_max_start)
                                if stem_max_start > 0
                                else 0
                            )
                            active_stems_with_offset.append(
                                (extra_stem, stem_window_start_ms)
                            )
                            base_instruments.add(extra_inst)
                            break

        # 4. オーディオとノートの読み込み・ミックス
        mixed_audio = np.zeros((2, self.window_frames), dtype=np.float32)
        note_groups = []

        for stem, stem_window_start_ms in active_stems_with_offset:
            stem_window_end_ms = stem_window_start_ms + self.window_ms
            wav_path = stem["wav_path"]
            # 指定された確率でリバーブ処理済みステム(stems_augments)に差し替える
            if rng.random() < self.p_use_stems_augments:
                wav_path = wav_path.replace("stems/", "stems_augments/").replace(
                    "stems\\", "stems_augments\\"
                )

            # 波形の読み込み
            audio = load_audio_window(
                wav_path,
                sample_rate=self.sample_rate,
                window_start_ms=stem_window_start_ms,
                window_ms=self.window_ms,
            )
            # データ拡張 (EQ, マイクロチューニング, リバーブ, ノイズ)
            if self.augmentor is not None and rng.random() < self.p_augment:
                audio = self.augmentor(audio)

            # 音量ランダム調整（-6dB 〜 6dB）
            gain = 10.0 ** (rng.uniform(-6.0, 6.0) / 20.0)
            mixed_audio += audio * gain

            # 事前処理済みのノートを読み込む
            with np.load(stem["npz_path"]) as data:
                start_ms = data["note_start_ms"]
                end_ms = data["note_end_ms"]
                pitch = data["note_pitch"]
                velocity = data["note_velocity"]
                instrument_ids = data.get("note_instrument", np.zeros_like(pitch))

            # 指定されたウィンドウに合わせてノートを切り出す
            carry_in, body = split_window_notes(
                start_ms=start_ms,
                end_ms=end_ms,
                pitch=pitch,
                velocity=velocity,
                instrument=instrument_ids,
                window_start_ms=stem_window_start_ms,
                window_end_ms=stem_window_end_ms,
                clip_note_end_to_window=True,
            )
            note_groups.extend([carry_in, body])

        # 5. ドラム耐性向上のためのランダムドラム追加
        has_drum = any("drum" in inst.lower() for inst in base_instruments)
        if not has_drum and self.drum_files and rng.random() < self.p_drum_mix:
            drum_path = rng.choice(self.drum_files)
            try:
                info = sf.info(drum_path)
                duration_ms = int(info.frames / info.samplerate * 1000)
                max_start = max(0, duration_ms - self.window_ms)
                drum_start_ms = rng.randint(0, max_start) if max_start > 0 else 0

                drum_audio = load_audio_window(
                    drum_path,
                    sample_rate=self.sample_rate,
                    window_start_ms=drum_start_ms,
                    window_ms=self.window_ms,
                )

                if self.augmentor is not None and rng.random() < self.p_augment:
                    drum_audio = self.augmentor(drum_audio)

                gain = 10.0 ** (rng.uniform(-6.0, 6.0) / 20.0)
                mixed_audio += drum_audio * gain
            except Exception as e:
                logger.warning(f"Failed to load drum file {drum_path}: {e}")

        # 加算ミックスによるクリッピングを防止
        peak = np.abs(mixed_audio).max()
        if peak > 1.0:
            mixed_audio /= peak

        audio_tensor = torch.from_numpy(mixed_audio).contiguous()

        # ノート情報をマージし、ターゲットラベルを生成
        merged_notes = concat_window_notes(*note_groups)

        frame_active_targets = build_frame_note_targets(
            active_start_ms=merged_notes.start_ms,
            active_end_ms=merged_notes.end_ms,
            active_pitch=merged_notes.pitch,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            num_frames=self.model_frames,
        )

        frame_instrument_targets = build_instrument_targets(
            active_start_ms=merged_notes.start_ms,
            active_end_ms=merged_notes.end_ms,
            active_pitch=merged_notes.pitch,
            active_instrument=merged_notes.instrument,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            num_frames=self.model_frames,
        )

        interval_targets = build_pitch_interval_targets(
            active_start_ms=merged_notes.start_ms,
            active_end_ms=merged_notes.end_ms,
            active_pitch=merged_notes.pitch,
            active_has_onset=merged_notes.has_onset,
            active_has_offset=merged_notes.has_offset,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            num_frames=self.model_frames,
        )

        # オーディオの有効長を計算 (ゼロ埋めされていない実際の長さ)
        max_valid_audio_ms = 0
        for stem, stem_window_start_ms in active_stems_with_offset:
            valid_ms = stem["duration_ms"] - stem_window_start_ms
            if valid_ms > max_valid_audio_ms:
                max_valid_audio_ms = valid_ms

        valid_audio_ms = max_valid_audio_ms
        if valid_audio_ms > self.window_ms:
            valid_audio_ms = self.window_ms
        if valid_audio_ms < 0:
            valid_audio_ms = 0
        valid_audio_frames_val = int(round(valid_audio_ms * self.sample_rate / 1000.0))

        # 楽器ラベルがないデータセットの場合は楽器分類ロスをマスクする
        mask_instrument_loss = selected_group.get("mask_instrument_loss", False)

        return {
            "song_name": song_name,
            "window_start_ms": window_start_ms,
            "audio": audio_tensor,
            "frame_active_targets": frame_active_targets,
            "frame_instrument_targets": frame_instrument_targets,
            "interval_targets": interval_targets,
            "valid_audio_frames": valid_audio_frames_val,
            "mask_instrument_loss": mask_instrument_loss,
        }
