from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from audio_emotion.load_models.download_model import download_model


_MODEL: Optional[Qwen2AudioForConditionalGeneration] = None
_PROCESSOR: Optional[AutoProcessor] = None
_INPUT_DEVICE: Optional[torch.device] = None


def _resolve_input_device(model: Qwen2AudioForConditionalGeneration) -> torch.device:
    """
    解决模型输入设备，优先使用模型的 hf_device_map 映射，如果没有则使用 cuda（如果可用）或 cpu。
    """
    if hasattr(model, "hf_device_map"):
        for mapped_device in model.hf_device_map.values():
            if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk", "meta"}:
                return torch.device(mapped_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_processor(
    force_reload: bool = False,
) -> tuple[Qwen2AudioForConditionalGeneration, AutoProcessor, torch.device]:
    """
    获取模型、处理器和输入设备，使用全局缓存以避免重复加载。
    如果 force_reload 为 True，则强制重新加载模型和处理器。
    """
    global _MODEL, _PROCESSOR, _INPUT_DEVICE

    if force_reload:
        _MODEL = None
        _PROCESSOR = None
        _INPUT_DEVICE = None

    if _MODEL is None or _PROCESSOR is None or _INPUT_DEVICE is None:
        model_dir = download_model()
        _PROCESSOR = AutoProcessor.from_pretrained(model_dir)
        _MODEL = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
        )
        _INPUT_DEVICE = _resolve_input_device(_MODEL)

    return _MODEL, _PROCESSOR, _INPUT_DEVICE


def get_model() -> Qwen2AudioForConditionalGeneration:
    model, _, _ = get_model_and_processor()
    return model


def get_processor() -> AutoProcessor:
    _, processor, _ = get_model_and_processor()
    return processor


def get_input_device() -> torch.device:
    _, _, input_device = get_model_and_processor()
    return input_device


__all__ = [
    "get_model",
    "get_processor",
    "get_input_device",
    "get_model_and_processor",
]
