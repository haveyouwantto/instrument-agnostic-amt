# @title Prepare stem-separated transcription helpers
from collections import defaultdict
from pathlib import Path
import shutil

import pretty_midi
import torch

import infer
from stem_splitter.inference import SeparationConfig, _separate_one_file, load_mss_model

from adtof_pytorch import transcribe_to_midi


INSTRUMENT_CLASS_GAIN: dict[str, int] = {
    # 打击乐与节奏组
    "drums": 88,                   # 鼓组会有很多高频的镲片声，要稍微调低一点
    "timpani": 115,                # 定音鼓是厚重的低音基石，要给它多多的力气
    "chromatic_percussion": 100,   # 钢片琴等音色比较纯净，保持中间位置
    "percussive_fx": 75,           # 敲击特效通常噪音多，要压低

    # 键盘类乐器
    "piano": 102,                  # 钢琴是全能选手，稍微抬高一点点保证清晰度
    "electric_piano": 100,         # 电钢琴声音通常比较圆润，保持中值
    "plucked_keyboard": 85,        # 拨弦古钢琴（羽管键琴）泛音太亮，要调低
    "organ": 74,                   # 管风琴是“泛音之王”，声音太厚了，必须压低
    "accordion_family": 78,        # 手风琴家族声音很扁且泛音多，调低一点
    "harmonica": 82,               # 口琴的高频比较尖锐，调低
    
    # 吉他类乐器（这里是高频噪音的“重灾区”喔）
    "acoustic_guitar": 82,         # 木吉他拨弦声很亮，要压低
    "electric_guitar_clean": 78,   # 清音电吉他也要温柔一点
    "electric_guitar_muted": 85,   # 闷音吉他高频被滤掉了，可以比清音稍微大一点
    "distorted_guitar": 65,        # 失真吉他全身都是噪点，必须狠狠压住
    "guitar_harmonics": 60,        # 这种纯粹的高频泛音最扎耳朵了，要最细小

    # 低音类乐器（泛音很少，需要抬高来撑场面）
    "acoustic_bass": 110,          # 原声贝斯需要很强的力量感
    "electric_bass": 108,          # 电贝斯也是音乐的底盘
    "slap_bass": 95,               # 勾击贝斯会有金属撞击声，比普通贝斯低一点
    "synth_bass": 98,              # 合成器贝斯通常自带很多修饰，稍微压一点点

    # 弦乐与管弦乐类
    "strings": 78,                 # 弦乐合奏泛音非常复杂，压低避免浑浊
    "pizzicato_strings": 88,       # 拨弦效果比拉弦清脆，可以稍大一点
    "orchestral_harp": 95,         # 竖琴很温柔，比钢琴小一点点就好
    "orchestra_hit": 60,           # 这种大合奏音效全是噪音，要调到非常低
    "choir": 72,                   # 合唱团是很多人声叠在一起，要非常小声才和谐
    "brass": 76,                   # 铜管乐器非常刺耳，要大幅调低
    "sax": 80,                     # 萨克斯风也是很有穿透力的，要压住
    "orchestral_woodwind": 88,     # 综合木管组通常比较柔和，中等偏下
    "flute_pipe": 110,             # 长笛声音最纯正（接近正弦波），要抬高

    # 合成器与特效
    "synth_lead": 76,              # 领奏合成器通常太尖，调低
    "synth_pad": 88,               # 铺底音色不能抢戏，调低
    "synth_fx": 70,                # 合成器特效大多是噪音，调低
    "ethnic": 85,                  # 民族乐器通常带有一些独特的共鸣，调低
    "sound_fx": 65,                # 纯粹的音效（比如开门声）通常很吵，调低

    "melody": 120,                 # 这是我们为 AMT 结果单独设定的“旋律”类
}


# セッション中にモデルを使い回して、再実行時の待ち時間を減らす。
STEM_PIPELINE_CACHE = {}


def merge_midis_logic(midi_paths, output_file, max_melodic=15):
    """ステムごとの MIDI を 1 本にまとめる。"""
    if not midi_paths:
        raise ValueError("No MIDI files to merge")

    # 1. 先頭 MIDI のテンポマップを土台として使う。
    master_pm = pretty_midi.PrettyMIDI(str(midi_paths[0]))

    all_notes = defaultdict(list)
    all_ccs = defaultdict(list)
    all_pbends = defaultdict(list)
    instrument_names = {}

    # 2. 同じ program / drum 属性ごとにノートを集約する。
    for path in midi_paths:
        pm = pretty_midi.PrettyMIDI(str(path))
        for inst in pm.instruments:
            if "drum" in path.stem.lower():
                inst.is_drum = True
            key = (inst.program, inst.is_drum)
            filtered_notes = [n for n in inst.notes if (n.end - n.start) < 15.0]
            all_notes[key].extend(filtered_notes)
            all_ccs[key].extend(inst.control_changes)
            all_pbends[key].extend(inst.pitch_bends)
            if key not in instrument_names:
                instrument_names[key] = inst.name

    melodic_keys = [k for k in all_notes.keys() if not k[1]]
    drum_keys = [k for k in all_notes.keys() if k[1]]
    melodic_keys.sort(key=lambda k: len(all_notes[k]), reverse=True)

    final_instruments = []
    if len(melodic_keys) > max_melodic:
        kept_keys = melodic_keys[: max_melodic - 1]
        overflow_keys = melodic_keys[max_melodic - 1 :]
        for key in kept_keys:
            inst = pretty_midi.Instrument(
                program=key[0],
                is_drum=key[1],
                name=instrument_names[key],
            )
            inst.notes = all_notes[key]
            inst.control_changes = all_ccs[key]
            inst.pitch_bends = all_pbends[key]
            final_instruments.append(inst)

        base_key = overflow_keys[0]
        overflow_inst = pretty_midi.Instrument(
            program=base_key[0],
            is_drum=base_key[1],
            name="Other / Merged",
        )
        for key in overflow_keys:
            overflow_inst.notes.extend(all_notes[key])
            overflow_inst.control_changes.extend(all_ccs[key])
            overflow_inst.pitch_bends.extend(all_pbends[key])
        final_instruments.append(overflow_inst)
    else:
        for key in melodic_keys:
            inst = pretty_midi.Instrument(
                program=key[0],
                is_drum=key[1],
                name=instrument_names[key],
            )
            inst.notes = all_notes[key]
            inst.control_changes = all_ccs[key]
            inst.pitch_bends = all_pbends[key]
            final_instruments.append(inst)

    for key in drum_keys:
        inst = pretty_midi.Instrument(
            program=key[0],
            is_drum=key[1],
            name=instrument_names[key],
        )
        inst.notes = all_notes[key]
        inst.control_changes = all_ccs[key]
        inst.pitch_bends = all_pbends[key]
        final_instruments.append(inst)

    master_pm.instruments = final_instruments
    for inst in master_pm.instruments:
        inst.notes.sort(key=lambda note: note.start)
    master_pm.write(str(output_file))


def resolve_stem_paths(*, song_path, stem_dir, stem_names):
    """既存 stem 出力の標準パスを組み立て、存在するものだけ返す。"""
    song_file = Path(song_path)
    song_id = song_file.stem
    stem_root = Path(stem_dir) / song_id
    resolved = {}

    # 1. 標準の保存規則から期待パスを直接構成する。
    for stem_name in stem_names:
        expected_path = stem_root / f"{song_id}_{stem_name}.wav"
        if expected_path.exists():
            resolved[stem_name] = expected_path

    if resolved:
        return resolved

    # 2. 名前が少しずれていても拾えるよう、末尾 stem 名で緩く補完する。
    if not stem_root.exists():
        return resolved

    for wav_path in sorted(stem_root.glob("*.wav")):
        stem_key = wav_path.stem
        for stem_name in stem_names:
            if stem_key.endswith(f"_{stem_name}"):
                resolved.setdefault(stem_name, wav_path)
                break

    return resolved


def get_stem_pipeline_models(checkpoint_path=None, device_preference=None, model_type="default"):
    """AMT とステム分離モデルを読み込み、セッション中は再利用する。"""
    device = torch.device(device_preference or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = infer._ensure_checkpoint(
        None if checkpoint_path in (None, "", "DEFAULT") else Path(checkpoint_path),
        model_type=model_type
    )
    amt_cache_key = ("amt", str(checkpoint.resolve()), device.type)
    sep_cache_key = ("sep", device.type)

    if sep_cache_key not in STEM_PIPELINE_CACHE:
        print(f"Loading Separation model on {device} ...")
        sep_config = SeparationConfig(skip_existing=True)
        sep_model = load_mss_model(sep_config, device=device)
        sep_dtype = torch.float16 if sep_config.use_half_precision and device.type == "cuda" else torch.float32
        STEM_PIPELINE_CACHE[sep_cache_key] = (sep_config, sep_model, sep_dtype)
    else:
        sep_config, sep_model, sep_dtype = STEM_PIPELINE_CACHE[sep_cache_key]

    if amt_cache_key not in STEM_PIPELINE_CACHE:
        print(f"Loading AMT model ({model_type}) on {device} ...")
        amt_model, amt_config, amt_settings = infer._load_model_and_settings(
            checkpoint,
            device=device,
            window_ms_override=None,
            stride_ms_override=None,
            track_batch_size_override=None,
        )
        STEM_PIPELINE_CACHE[amt_cache_key] = (amt_model, amt_config, amt_settings)
    else:
        print(f"Reusing cached AMT model ({model_type}) on {device} ...")
        amt_model, amt_config, amt_settings = STEM_PIPELINE_CACHE[amt_cache_key]

    return {
        "device": device,
        "checkpoint": checkpoint,
        "amt_model": amt_model,
        "amt_config": amt_config,
        "amt_settings": amt_settings,
        "sep_config": sep_config,
        "sep_model": sep_model,
        "sep_dtype": sep_dtype,
    }


def run_stem_separated_transcription(
    audio_path,
    *,
    checkpoint_path=None,
    output_root="colab_outputs",
    window_batch_size=4,
    max_midi_melodic_instruments=15,
    skip_drum_stems=True,
    cleanup_separated_stems=False,
    merge_onset_ms=20.0,
):
    """ステム分離 -> 各ステム採譜 -> MIDI マージを 1 回で実行する。"""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    bundle = get_stem_pipeline_models(checkpoint_path=checkpoint_path, model_type="default")
    device = bundle["device"]
    amt_model = bundle["amt_model"]
    amt_config = bundle["amt_config"]
    amt_settings = bundle["amt_settings"]
    sep_config = bundle["sep_config"]
    sep_model = bundle["sep_model"]
    sep_dtype = bundle["sep_dtype"]

    bass_bundle = get_stem_pipeline_models(checkpoint_path=checkpoint_path, model_type="bass")
    bass_amt_model = bass_bundle["amt_model"]
    bass_amt_config = bass_bundle["amt_config"]
    bass_amt_settings = bass_bundle["amt_settings"]

    vocal_bundle = get_stem_pipeline_models(checkpoint_path=checkpoint_path, model_type="vocal")
    vocal_amt_model = vocal_bundle["amt_model"]
    vocal_amt_config = vocal_bundle["amt_config"]
    vocal_amt_settings = vocal_bundle["amt_settings"]

    guitar_bundle = get_stem_pipeline_models(checkpoint_path=checkpoint_path, model_type="guitar")
    guitar_amt_model = guitar_bundle["amt_model"]
    guitar_amt_config = guitar_bundle["amt_config"]
    guitar_amt_settings = guitar_bundle["amt_settings"]

    # 1. 出力先を曲ごとに分ける。
    run_root = Path(output_root) / audio_file.stem
    stem_dir = run_root / "stems"
    stem_midi_dir = run_root / "stem_midis"
    merged_dir = run_root / "merged"
    for directory in (stem_dir, stem_midi_dir, merged_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 2. まず元音源をステム分離する。
    print(f"Separating stems for: {audio_file.name}")
    stems = _separate_one_file(
        audio_file,
        stem_dir,
        sep_config,
        sep_model,
        device,
        sep_dtype,
    )

    # 3. 再実行時に分離が省略された場合は、既存 stem の標準パスを復元する。
    if not stems:
        stems = resolve_stem_paths(
            song_path=audio_file,
            stem_dir=stem_dir,
            stem_names=sep_config.stem_names,
        )
        if stems:
            print(f"Reusing existing stems: {sorted(stems)}")
        else:
            raise RuntimeError(f"No stems found for {audio_file.stem}")

    # 4. ドラム以外の各ステムを順に採譜する。
    song_midi_paths = []
    for stem_name, stem_path in sorted(stems.items()):

        output_midi = stem_midi_dir / f"{audio_file.stem}_{stem_name}.mid"
        print(f"Transcribing stem: {stem_name}")
        if skip_drum_stems and "drum" in stem_name.lower():
            print(f"Transcribe drum stem using ADTOF: {stem_name}")
            transcribe_to_midi(stem_path, output_midi)
            song_midi_paths.append(output_midi)
            continue

        if "bass" in stem_name.lower():
            current_amt_model = bass_amt_model
            current_amt_config = bass_amt_config
            current_amt_settings = bass_amt_settings
        elif "vocal" in stem_name.lower():
            current_amt_model = vocal_amt_model
            current_amt_config = vocal_amt_config
            current_amt_settings = vocal_amt_settings
        elif "guitar" in stem_name.lower():
            current_amt_model = guitar_amt_model
            current_amt_config = guitar_amt_config
            current_amt_settings = guitar_amt_settings
        else:
            current_amt_model = amt_model
            current_amt_config = amt_config
            current_amt_settings = amt_settings

        waveform, _, _ = infer._load_audio(
            Path(stem_path),
            target_sample_rate=current_amt_config.sample_rate,
        )
        notes, _, _ = infer.run_inference(
            model=current_amt_model,
            waveform=waveform.to(device),
            model_config=current_amt_config,
            settings=current_amt_settings,
            device=device,
            amp_enabled=False,
            amp_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            velocity=100,
            merge_gap_ms=None,
            merge_onset_ms=merge_onset_ms,
            silence_gate_rms_dbfs=-72,
            window_batch_size=window_batch_size,
            max_midi_melodic_instruments=max_midi_melodic_instruments,
            disable_tqdm=True,
            max_note_seconds=15.0
        )

        midi = infer._build_midi(notes, sample_rate=current_amt_config.sample_rate, instrument_volumes=INSTRUMENT_CLASS_GAIN)
        midi.write(str(output_midi))
        song_midi_paths.append(output_midi)

    if not song_midi_paths:
        raise RuntimeError("No stem MIDI files were generated")

    # 5. ステムごとの MIDI を 1 本にまとめる。
    merged_midi_path = merged_dir / f"{audio_file.stem}.mid"
    merge_midis_logic(
        song_midi_paths,
        merged_midi_path,
        max_melodic=max_midi_melodic_instruments,
    )

    # 6. 必要なら中間の分離 wav を消して容量を節約する。
    if cleanup_separated_stems:
        shutil.rmtree(stem_dir, ignore_errors=True)

    result = {
        "audio_path": str(audio_file),
        "output_root": str(run_root),
        "stem_dir": str(stem_dir),
        "stem_midi_dir": str(stem_midi_dir),
        "merged_midi_path": str(merged_midi_path),
        "stem_count": len(stems),
        "transcribed_stem_count": len(song_midi_paths),
    }
    print("Merged MIDI:", merged_midi_path)
    return result

