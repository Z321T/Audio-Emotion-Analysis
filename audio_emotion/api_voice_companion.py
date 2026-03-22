from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .emollm_service import generate_companion_reply
from .load import (
    release_audio_model_and_processor,
    release_emollm_model_and_tokenizer,
    release_tts_model,
)
from .qwen_audio_service import analyze_audio_waveform
from .tts_service import play_audio_blocking, synthesize_reply_audio


DEFAULT_AUDIO_SAMPLE_RATE = 16000


async def voice_companion_pipeline(
    audio_path: str | Path,
    *args: Any,
    play_audio: bool = True,
    save_reply_audio_path: str | Path | None = None,
    max_audio_tokens: int = 256,
    max_reply_tokens: int = 256,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    运行完整语音陪护链路（文件音频输入）：音频分析 -> 文本回复 -> 语音合成。

    参数：
        audio_path: 用户输入音频路径。
        play_audio: 是否立即播放合成语音，默认 True。
        save_reply_audio_path: 是否保存回复音频到本地路径，默认不保存。
        max_audio_tokens: 音频分析阶段最大生成 token 数。
        max_reply_tokens: 回复生成阶段最大生成 token 数。
        *args: 预留可变参数，便于后续扩展。
        **kwargs: 预留关键字参数，便于后续扩展。

    返回：
        返回完整链路结果字典：
        {
            "analysis": {
                "transcription": "音频中文转写结果",
                "emotion": "整体情绪标签"
            },
            "reply_text": "陪护模型回复文本",
            "reply_audio_path": "可选，保存的音频路径字符串"
        }
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"未找到音频文件: {path}")

    sampling_rate = DEFAULT_AUDIO_SAMPLE_RATE
    audio_waveform, _ = await asyncio.to_thread(librosa.load, path, sr=sampling_rate, mono=True)

    print("[步骤1] Qwen2-Audio：开始音频分析")
    try:
        analysis = await asyncio.to_thread(
            analyze_audio_waveform,
            audio_waveform,
            str(path),
            max_audio_tokens,
        )
    finally:
        await asyncio.to_thread(release_audio_model_and_processor)
        print("[步骤1] 已释放 Qwen2-Audio 模型")
    print(f"[步骤1] 完成，转写={analysis.get('transcription', '')}")
    print(f"[步骤1] 完成，情绪={analysis.get('emotion', '未知')}")

    print("[步骤2] EmoLLM：开始生成陪护回复")
    try:
        reply_text = await asyncio.to_thread(
            generate_companion_reply,
            analysis.get("transcription", ""),
            analysis.get("emotion", "未知"),
            max_reply_tokens,
        )
    finally:
        await asyncio.to_thread(release_emollm_model_and_tokenizer)
        print("[步骤2] 已释放 EmoLLM 模型")
    print(f"[步骤2] 完成，回复={reply_text}")

    print("[步骤3] Qwen3-TTS-12Hz：开始语音合成")
    try:
        reply_wav, reply_sr = await asyncio.to_thread(
            synthesize_reply_audio,
            reply_text,
            (audio_waveform, sampling_rate),
        )
    finally:
        await asyncio.to_thread(release_tts_model)
        print("[步骤3] 已释放 Qwen3-TTS-12Hz 模型")
    print("[步骤3] 完成，语音合成成功")

    saved_path: str | None = None
    if save_reply_audio_path is not None:
        out_path = Path(save_reply_audio_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(sf.write, out_path, reply_wav, reply_sr)
        saved_path = str(out_path)
        print(f"[步骤3] 已保存语音到: {saved_path}")

    if play_audio:
        print("[步骤3] 开始播放语音")
        await asyncio.to_thread(play_audio_blocking, reply_wav, reply_sr)
        print("[步骤3] 播放结束")

    return {
        "analysis": analysis,
        "reply_text": reply_text,
        "reply_audio_path": saved_path,
    }


async def voice_companion_pipeline_from_waveform(
    audio_waveform: np.ndarray,
    sample_rate: int,
    source_tag: str,
    *args: Any,
    play_audio: bool = True,
    max_audio_tokens: int = 256,
    max_reply_tokens: int = 256,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    运行完整语音陪护链路（内存波形输入）：音频分析 -> 文本回复 -> 语音合成。

    参数：
        audio_waveform: 用户输入音频波形（一维数组）。
        sample_rate: 波形采样率。
        source_tag: 当前音频来源标识（例如 microphone_turn_1.wav）。
        play_audio: 是否立即播放合成语音，默认 True。
        max_audio_tokens: 音频分析阶段最大生成 token 数。
        max_reply_tokens: 回复生成阶段最大生成 token 数。
        *args: 预留可变参数，便于后续扩展。
        **kwargs: 预留关键字参数，便于后续扩展。

    返回：
        返回完整链路结果字典：
        {
            "analysis": {
                "transcription": "音频中文转写结果",
                "emotion": "整体情绪标签"
            },
            "reply_text": "陪护模型回复文本"
        }
    """
    print("[步骤1] Qwen2-Audio：开始音频分析")
    try:
        analysis = await asyncio.to_thread(
            analyze_audio_waveform,
            audio_waveform,
            source_tag,
            max_audio_tokens,
        )
    finally:
        await asyncio.to_thread(release_audio_model_and_processor)
        print("[步骤1] 已释放 Qwen2-Audio 模型")
    print(f"[步骤1] 完成，转写={analysis.get('transcription', '')}")
    print(f"[步骤1] 完成，情绪={analysis.get('emotion', '未知')}")

    print("[步骤2] EmoLLM：开始生成陪护回复")
    try:
        reply_text = await asyncio.to_thread(
            generate_companion_reply,
            analysis.get("transcription", ""),
            analysis.get("emotion", "未知"),
            max_reply_tokens,
        )
    finally:
        await asyncio.to_thread(release_emollm_model_and_tokenizer)
        print("[步骤2] 已释放 EmoLLM 模型")
    print(f"[步骤2] 完成，回复={reply_text}")

    print("[步骤3] Qwen3-TTS-12Hz：开始语音合成")
    try:
        reply_wav, reply_sr = await asyncio.to_thread(
            synthesize_reply_audio,
            reply_text,
            (audio_waveform, sample_rate),
        )
    finally:
        await asyncio.to_thread(release_tts_model)
        print("[步骤3] 已释放 Qwen3-TTS-12Hz 模型")
    print("[步骤3] 完成，语音合成成功")

    if play_audio:
        print("[步骤3] 开始播放语音")
        await asyncio.to_thread(play_audio_blocking, reply_wav, reply_sr)
        print("[步骤3] 播放结束")

    return {
        "analysis": analysis,
        "reply_text": reply_text,
    }
