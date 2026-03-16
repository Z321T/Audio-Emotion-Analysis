from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from audio_emotion.load import get_model_and_processor


DEFAULT_ANALYSIS_PROMPT = (
    "请先给出音频中的完整中文转写，再判断整体情绪。"
    "仅输出 JSON，格式必须为: "
    '{"transcription":"...","emotion":"...","reason":"..."}'
)


def _analyze_audio_waveform(
    audio_waveform: np.ndarray,
    source_tag: str,
    max_new_tokens: int = 256,
    prompt: str = DEFAULT_ANALYSIS_PROMPT,
) -> str:
    model, processor, input_device = get_model_and_processor()

    conversation: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "你是一名专业的音频内容与情绪分析助手。请始终使用中文回答。",
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": source_tag},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=text, audio=audio_waveform, return_tensors="pt", padding=True)

    model_inputs = {
        key: value.to(input_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    prompt_len = model_inputs["input_ids"].shape[1]
    generated = generated[:, prompt_len:]

    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def analyze_audio_file(audio_path: str | Path, max_new_tokens: int = 256) -> str:
    """
    输入本地音频文件路径，返回中文分析结果（转写 + 情绪）。
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"未找到音频文件: {path}")

    _, processor, _ = get_model_and_processor()
    sampling_rate = processor.feature_extractor.sampling_rate
    audio_waveform, _ = librosa.load(path, sr=sampling_rate, mono=True)

    return _analyze_audio_waveform(
        audio_waveform=audio_waveform,
        source_tag=str(path),
        max_new_tokens=max_new_tokens,
    )


def _record_microphone_audio(
    sample_rate: int,
    max_seconds: int = 30,
) -> np.ndarray:
    if max_seconds > 30:
        max_seconds = 30

    try:
        import msvcrt
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "麦克风对话需要安装 sounddevice，先执行: pip install sounddevice"
        ) from exc

    print("请按回车开始本轮录音，录音中再次按回车可提前结束。")
    input()

    frames: list[np.ndarray] = []

    def _callback(indata: np.ndarray, frames_count: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"录音状态: {status}")
        frames.append(indata.copy())

    start = time.time()
    print(f"开始录音，最长 {max_seconds} 秒。")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=_callback):
        while True:
            elapsed = time.time() - start
            if elapsed >= max_seconds:
                print("已达到 30 秒上限，停止录音并开始模型处理。")
                break
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in {"\r", "\n"}:
                    break
            time.sleep(0.05)

    if not frames:
        raise RuntimeError("未采集到音频，请检查麦克风设备。")

    audio_waveform = np.concatenate(frames, axis=0).squeeze(axis=-1)
    return audio_waveform


def interactive_microphone_dialog(
    max_seconds_per_turn: int = 30,
    max_new_tokens: int = 256,
) -> None:
    """
    伪流式麦克风对话：每轮录音（<=30秒）-> 推理 -> 输出 -> 等待下一轮。
    手动输入 q 可结束程序。
    """
    _, processor, _ = get_model_and_processor()
    sampling_rate = processor.feature_extractor.sampling_rate

    print("已进入麦克风对话模式。")
    print("说明: 单轮音频建议 30 秒以内；超时会自动截断并立即处理。")

    turn = 1
    while True:
        command = input("输入 q 退出，直接回车将准备开始第 %d 轮录音: " % turn).strip().lower()
        if command == "q":
            print("已退出麦克风对话模式。")
            break

        try:
            audio_waveform = _record_microphone_audio(
                sample_rate=sampling_rate,
                max_seconds=max_seconds_per_turn,
            )
            response = _analyze_audio_waveform(
                audio_waveform=audio_waveform,
                source_tag=f"microphone_turn_{turn}.wav",
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            print(f"第 {turn} 轮处理失败: {exc}")
            continue

        print(f"第 {turn} 轮模型回答:\n{response}\n")
        turn += 1


if __name__ == "__main__":
    # 示例：直接运行 inference.py 后进入麦克风轮次对话
    interactive_microphone_dialog()
