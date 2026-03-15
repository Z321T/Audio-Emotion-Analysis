from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from audio_emotion.load_models.download_model import download_model


_MODEL: Qwen2AudioForConditionalGeneration | None = None
_PROCESSOR: AutoProcessor | None = None


def _resolve_input_device(model: Qwen2AudioForConditionalGeneration) -> torch.device:
    """
    根据模型的 device_map 自动推断输入数据应该放在哪个设备上。
    优先使用模型的 hf_device_map 中的设备（如果存在且有效），否则根据当前环境自动选择 CUDA 或 CPU。
    """
    if hasattr(model, "hf_device_map"):
        for mapped_device in model.hf_device_map.values():
            if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk", "meta"}:
                return torch.device(mapped_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model_and_processor() -> tuple[Qwen2AudioForConditionalGeneration, AutoProcessor]:
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        model_dir = download_model()
        _PROCESSOR = AutoProcessor.from_pretrained(model_dir)
        _MODEL = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
        )
    return _MODEL, _PROCESSOR


def analyze_audio_file(audio_path: str | Path, max_new_tokens: int = 256) -> str:
    """
    输入本地音频文件路径，返回中文分析结果（转写 + 情绪）。
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"未找到音频文件: {path}")

    model, processor = _get_model_and_processor()
    sampling_rate = processor.feature_extractor.sampling_rate
    audio_waveform, _ = librosa.load(path, sr=sampling_rate, mono=True)

    conversation: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "你是一名专业的音频内容与情绪分析助手。请始终使用中文回答。",
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(path)},
                {
                    "type": "text",
                    "text": (
                        "请先给出音频中的完整中文转写，再判断整体情绪。"
                        "仅输出 JSON，格式必须为: "
                        '{"transcription":"...","emotion":"...","reason":"..."}'
                    ),
                },
            ],
        },
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=text, audio=audio_waveform, return_tensors="pt", padding=True)

    input_device = _resolve_input_device(model)
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
