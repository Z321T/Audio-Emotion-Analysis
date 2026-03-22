from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from .load import get_audio_model_and_processor


DEFAULT_ANALYSIS_PROMPT = (
    "请先给出音频中的完整中文转写，再判断整体情绪。"
    "仅输出 JSON，格式必须为: "
    '{"transcription":"...","emotion":"..."}'
)


def normalize_analysis_response(raw_text: str) -> dict[str, str]:
    """
    将音频模型输出文本标准化为仅包含转写和情绪的字典结构。

    参数：
        raw_text: 音频模型原始文本输出，预期为 JSON 字符串，但可能包含额外前后缀文本。

    返回：
        返回标准化字典：
        {
            "transcription": "音频中文转写结果",
            "emotion": "整体情绪标签"
        }
        若解析失败，会使用回退逻辑将原始文本写入 `transcription` 并将情绪置为“未知”。
    """
    cleaned = raw_text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        parsed = json.loads(cleaned)
        transcription = str(parsed.get("transcription", "")).strip()
        emotion = str(parsed.get("emotion", "")).strip()
        if not transcription:
            transcription = "未识别到有效语音内容"
        if not emotion:
            emotion = "未知"
        return {"transcription": transcription, "emotion": emotion}
    except Exception:
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, dict):
                transcription = str(parsed.get("transcription", "")).strip()
                emotion = str(parsed.get("emotion", "")).strip()
                if not transcription:
                    transcription = "未识别到有效语音内容"
                if not emotion:
                    emotion = "未知"
                return {"transcription": transcription, "emotion": emotion}
        except Exception:
            pass

        fallback = raw_text.strip() or "未识别到有效语音内容"
        return {"transcription": fallback, "emotion": "未知"}


def analyze_audio_waveform(
    audio_waveform: np.ndarray,
    source_tag: str,
    max_new_tokens: int = 256,
    prompt: str = DEFAULT_ANALYSIS_PROMPT,
) -> dict[str, str]:
    """
    对单段音频波形执行转写与情绪识别，返回标准化结果。

    参数：
        audio_waveform: 单通道音频波形，类型为 `np.ndarray`。
        source_tag: 当前音频来源标识，用于构建多模态对话内容（可为路径或轮次标签）。
        max_new_tokens: 生成最大 token 数，用于控制输出长度。
        prompt: 任务提示词，要求模型输出仅含 `transcription` 与 `emotion` 的 JSON。

    返回：
        返回标准化字典：
        {
            "transcription": "音频中文转写结果",
            "emotion": "整体情绪标签"
        }
    """
    model, processor, input_device = get_audio_model_and_processor()
    model_dtype = getattr(model, "dtype", None)

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
    inputs = processor(
        text=text,
        audio=audio_waveform,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    model_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            model_inputs[key] = value
            continue
        moved_value = value.to(input_device)
        if model_dtype is not None and torch.is_tensor(moved_value) and moved_value.is_floating_point():
            moved_value = moved_value.to(model_dtype)
        model_inputs[key] = moved_value

    with torch.inference_mode():
        generated = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    prompt_len = model_inputs["input_ids"].shape[1]
    generated = generated[:, prompt_len:]

    raw_text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return normalize_analysis_response(raw_text)


def analyze_audio_file(audio_path: str | Path, max_new_tokens: int = 256) -> str:
    """
    输入本地音频文件并返回标准 JSON 字符串（转写 + 情绪）。

    参数：
        audio_path: 本地音频文件路径，支持 `str` 与 `Path`。
        max_new_tokens: 音频分析阶段最大生成 token 数。

    返回：
        返回 JSON 字符串，结构为：
        {
            "transcription": "音频中文转写结果",
            "emotion": "整体情绪标签"
        }
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"未找到音频文件: {path}")

    _, processor, _ = get_audio_model_and_processor()
    sampling_rate = processor.feature_extractor.sampling_rate
    audio_waveform, _ = librosa.load(path, sr=sampling_rate, mono=True)

    result = analyze_audio_waveform(
        audio_waveform=audio_waveform,
        source_tag=str(path),
        max_new_tokens=max_new_tokens,
    )
    return json.dumps(result, ensure_ascii=False)


async def analyze_audio_file_async(audio_path: str | Path, max_new_tokens: int = 256) -> dict[str, str]:
    """
    异步分析本地音频文件，输出转写与情绪字典。

    参数：
        audio_path: 本地音频文件路径，支持 `str` 与 `Path`。
        max_new_tokens: 音频分析阶段最大生成 token 数。

    返回：
        返回字典结构：
        {
            "transcription": "音频中文转写结果",
            "emotion": "整体情绪标签"
        }
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"未找到音频文件: {path}")

    _, processor, _ = get_audio_model_and_processor()
    sampling_rate = processor.feature_extractor.sampling_rate
    audio_waveform, _ = await asyncio.to_thread(librosa.load, path, sr=sampling_rate, mono=True)
    return await asyncio.to_thread(
        analyze_audio_waveform,
        audio_waveform,
        str(path),
        max_new_tokens,
        DEFAULT_ANALYSIS_PROMPT,
    )
