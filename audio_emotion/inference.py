from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np

from .api_voice_companion import voice_companion_pipeline_from_waveform
from .qwen_audio_service import analyze_audio_file, analyze_audio_file_async


DEFAULT_MIC_SAMPLE_RATE = 16000


def _record_microphone_audio(
    sample_rate: int,
    max_seconds: int = 30,
    min_seconds: float = 1.2,
    silence_threshold: float = 0.008,
    silence_duration: float = 1.0,
) -> np.ndarray:
    """
    从麦克风录制一轮用户语音，检测静音后自动结束。

    参数：
        sample_rate: 录音采样率。
        max_seconds: 单轮最大录音时长，超过后会自动停止。
        min_seconds: 单轮最短录音时长，避免过早截断。
        silence_threshold: 静音判定阈值（基于 RMS）。
        silence_duration: 连续静音达到该时长后结束录音。

    返回：
        返回一维 `np.ndarray` 音频波形。
        若未录到有效语音或麦克风不可用，会直接抛出异常。
    """
    max_seconds = min(max_seconds, 30)

    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "麦克风对话需要安装 sounddevice，先执行: pip install sounddevice"
        ) from exc

    frames: list[np.ndarray] = []
    block_size = max(1, int(sample_rate * 0.2))
    started = False
    last_voice_time = time.time()
    start_time = time.time()

    print(f"开始录音，最长 {max_seconds} 秒。请直接说话。")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_size) as stream:
        while True:
            block, _ = stream.read(block_size)
            frames.append(block.copy())

            rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))
            now = time.time()
            elapsed = now - start_time

            if rms >= silence_threshold:
                started = True
                last_voice_time = now

            if elapsed >= max_seconds:
                print("达到本轮录音上限，停止录音。")
                break

            if started and elapsed >= min_seconds and (now - last_voice_time) >= silence_duration:
                print("检测到静音，停止录音。")
                break

    if not frames:
        raise RuntimeError("未采集到音频，请检查麦克风设备。")

    audio_waveform = np.concatenate(frames, axis=0).squeeze(axis=-1)
    if np.max(np.abs(audio_waveform)) < 1e-4:
        raise RuntimeError("未检测到有效语音，请检查麦克风输入音量。")
    return audio_waveform


async def run_voice_companion_dialog(
    max_seconds_per_turn: int = 30,
    max_audio_tokens: int = 256,
    max_reply_tokens: int = 256,
    max_turns: int | None = None,
) -> None:
    """
    运行完整语音陪护流程：麦克风输入 -> 音频分析 -> EmoLLM回复 -> TTS合成并播放。

    参数：
        max_seconds_per_turn: 单轮录音最大时长（秒），最大不超过 30 秒。
        max_audio_tokens: 音频分析阶段最大生成 token 数。
        max_reply_tokens: 陪护回复阶段最大生成 token 数。
        max_turns: 最大轮数，`None` 表示无限轮，需通过 `Ctrl+C` 主动结束。

    返回：
        无。该函数会持续执行对话循环直到达到轮次上限或被用户中断。
    """
    sampling_rate = DEFAULT_MIC_SAMPLE_RATE

    print("已进入语音陪护模式。")
    print("流程：自动录音 -> 情绪分析 -> 文本回复 -> 语音播报。按 Ctrl+C 结束。")

    turn = 1
    while True:
        if max_turns is not None and turn > max_turns:
            print("达到设定轮数，已结束语音陪护。")
            break

        try:
            print(f"\n--- 第 {turn} 轮 ---")
            audio_waveform = await asyncio.to_thread(
                _record_microphone_audio,
                sample_rate=sampling_rate,
                max_seconds=max_seconds_per_turn,
            )

            _ = await voice_companion_pipeline_from_waveform(
                audio_waveform=audio_waveform,
                sample_rate=sampling_rate,
                source_tag=f"microphone_turn_{turn}.wav",
                play_audio=True,
                max_audio_tokens=max_audio_tokens,
                max_reply_tokens=max_reply_tokens,
            )
        except Exception as exc:
            print(f"第 {turn} 轮处理失败: {exc}")
            continue
        turn += 1


def interactive_microphone_dialog(
    max_seconds_per_turn: int = 30,
    max_new_tokens: int = 256,
) -> None:
    """
    兼容旧接口并启动自动语音陪护循环。

    参数：
        max_seconds_per_turn: 单轮录音最大时长（秒）。
        max_new_tokens: 兼容参数，映射到音频分析与文本回复的生成长度。

    返回：
        无。内部通过 `asyncio.run` 执行异步陪护循环。
    """
    asyncio.run(
        run_voice_companion_dialog(
            max_seconds_per_turn=max_seconds_per_turn,
            max_audio_tokens=max_new_tokens,
            max_reply_tokens=max_new_tokens,
            max_turns=None,
        )
    )


__all__ = [
    "analyze_audio_file",
    "analyze_audio_file_async",
    "interactive_microphone_dialog",
    "run_voice_companion_dialog",
]


if __name__ == "__main__":
    interactive_microphone_dialog()
