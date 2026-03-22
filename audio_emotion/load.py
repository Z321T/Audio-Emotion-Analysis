from __future__ import annotations

import asyncio
import gc
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2AudioForConditionalGeneration,
)

from .load_models.download_model import download_model
from .qwen_tts import Qwen3TTSModel


_AUDIO_MODEL: Optional[Qwen2AudioForConditionalGeneration] = None
_AUDIO_PROCESSOR: Optional[AutoProcessor] = None
_AUDIO_INPUT_DEVICE: Optional[torch.device] = None

_EMOLLM_MODEL: Optional[PreTrainedModel] = None
_EMOLLM_TOKENIZER: Optional[PreTrainedTokenizerBase] = None
_EMOLLM_INPUT_DEVICE: Optional[torch.device] = None

_TTS_MODEL: Optional[Qwen3TTSModel] = None

QWEN2_AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
EMOLLM_MODEL_ID = "aJupyter/EmoLLM_Qwen2-7B-Instruct_lora"
QWEN_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def _resolve_input_device(model: PreTrainedModel) -> torch.device:
    """
    解析模型输入所在设备，优先读取 `hf_device_map`，否则回退到模型参数设备。 

    参数：
        model: 已加载完成的 HuggingFace 预训练模型实例。

    返回：
        返回可用于输入张量迁移的 `torch.device`。
    """
    if hasattr(model, "hf_device_map"):
        for mapped_device in model.hf_device_map.values():
            if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk", "meta"}:
                return torch.device(mapped_device)
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cleanup_memory() -> None:
    """
    统一执行 Python 与 CUDA 显存清理，减少大模型切换时的内存残留。

    参数：
        无。

    返回：
        无。该函数仅执行清理动作，不返回业务结果。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_audio_model_and_processor(
    force_reload: bool = False,
) -> tuple[Qwen2AudioForConditionalGeneration, AutoProcessor, torch.device]:
    """
    获取音频理解模型、处理器和输入设备，默认使用单例缓存避免重复加载。 

    参数：
        force_reload: 是否强制重新加载模型和处理器。为 `True` 时会清空缓存并重新初始化。

    返回：
        返回三元组 `(audio_model, audio_processor, input_device)`：
        - `audio_model`: Qwen2-Audio 模型实例
        - `audio_processor`: 音频处理器
        - `input_device`: 推理输入张量应迁移到的设备
    """
    global _AUDIO_MODEL, _AUDIO_PROCESSOR, _AUDIO_INPUT_DEVICE

    if force_reload:
        _AUDIO_MODEL = None
        _AUDIO_PROCESSOR = None
        _AUDIO_INPUT_DEVICE = None

    if _AUDIO_MODEL is None or _AUDIO_PROCESSOR is None or _AUDIO_INPUT_DEVICE is None:
        model_dir = download_model(QWEN2_AUDIO_MODEL_ID)
        _AUDIO_PROCESSOR = AutoProcessor.from_pretrained(model_dir)
        _AUDIO_MODEL = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        _AUDIO_INPUT_DEVICE = _resolve_input_device(_AUDIO_MODEL)

    return _AUDIO_MODEL, _AUDIO_PROCESSOR, _AUDIO_INPUT_DEVICE


def get_emollm_model_and_tokenizer(
    force_reload: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    """
    获取 EmoLLM 对话模型、分词器与输入设备，支持本地单例复用。 

    参数：
        force_reload: 是否强制重新加载 EmoLLM。为 `True` 时会清空缓存并重新初始化。

    返回：
        返回三元组 `(chat_model, chat_tokenizer, input_device)`：
        - `chat_model`: EmoLLM 因果语言模型实例
        - `chat_tokenizer`: 对应分词器
        - `input_device`: 文本输入张量应迁移到的设备
    """
    global _EMOLLM_MODEL, _EMOLLM_TOKENIZER, _EMOLLM_INPUT_DEVICE

    if force_reload:
        _EMOLLM_MODEL = None
        _EMOLLM_TOKENIZER = None
        _EMOLLM_INPUT_DEVICE = None

    if _EMOLLM_MODEL is None or _EMOLLM_TOKENIZER is None or _EMOLLM_INPUT_DEVICE is None:
        model_dir = download_model(EMOLLM_MODEL_ID)
        _EMOLLM_TOKENIZER = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        _EMOLLM_MODEL = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        _EMOLLM_INPUT_DEVICE = _resolve_input_device(_EMOLLM_MODEL)

    return _EMOLLM_MODEL, _EMOLLM_TOKENIZER, _EMOLLM_INPUT_DEVICE


def get_tts_model(force_reload: bool = False) -> Qwen3TTSModel:
    """
    获取 Qwen3-TTS 模型实例并复用缓存，避免重复初始化。 

    参数：
        force_reload: 是否强制重新加载 TTS 模型。为 `True` 时会清空缓存并重新初始化。

    返回：
        返回可直接调用的 `Qwen3TTSModel` 实例。
    """
    global _TTS_MODEL

    if force_reload:
        _TTS_MODEL = None

    if _TTS_MODEL is None:
        model_dir = download_model(QWEN_TTS_MODEL_ID)
        # 当前环境中 Qwen3-TTS 的部分路径存在 bf16/fp16 混用，统一为 fp32 以保证稳定性。
        dtype = torch.float32
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        _TTS_MODEL = Qwen3TTSModel.from_pretrained(
            str(model_dir),
            device_map=device_map,
            dtype=dtype,
        )

    return _TTS_MODEL


def release_audio_model_and_processor() -> None:
    """
    释放 Qwen2-Audio 模型、处理器与输入设备缓存，供低显存场景按步骤回收。

    参数：
        无。

    返回：
        无。释放后再次调用音频分析会自动重新加载模型。
    """
    global _AUDIO_MODEL, _AUDIO_PROCESSOR, _AUDIO_INPUT_DEVICE
    _AUDIO_MODEL = None
    _AUDIO_PROCESSOR = None
    _AUDIO_INPUT_DEVICE = None
    _cleanup_memory()


def release_emollm_model_and_tokenizer() -> None:
    """
    释放 EmoLLM 模型、分词器与输入设备缓存，供低显存场景按步骤回收。

    参数：
        无。

    返回：
        无。释放后再次调用文本回复会自动重新加载模型。
    """
    global _EMOLLM_MODEL, _EMOLLM_TOKENIZER, _EMOLLM_INPUT_DEVICE
    _EMOLLM_MODEL = None
    _EMOLLM_TOKENIZER = None
    _EMOLLM_INPUT_DEVICE = None
    _cleanup_memory()


def release_tts_model() -> None:
    """
    释放 Qwen3-TTS 模型缓存，供低显存场景在语音合成后立即回收。

    参数：
        无。

    返回：
        无。释放后再次调用 TTS 会自动重新加载模型。
    """
    global _TTS_MODEL
    _TTS_MODEL = None
    _cleanup_memory()


def release_all_models() -> None:
    """
    一次性释放三类模型缓存（音频分析、EmoLLM、TTS），用于手动全量回收。

    参数：
        无。

    返回：
        无。释放后所有模型会在下一次调用对应步骤时重新加载。
    """
    release_audio_model_and_processor()
    release_emollm_model_and_tokenizer()
    release_tts_model()


async def warmup_all_models(force_reload: bool = False) -> None:
    """
    异步预热三类模型（音频分析、EmoLLM、TTS），用于服务启动阶段提前加载。 

    参数：
        force_reload: 是否强制重新加载全部模型。为 `True` 时会跳过缓存并完整重建。

    返回：
        无。若模型加载失败会直接抛出异常。
    """
    await asyncio.gather(
        asyncio.to_thread(get_audio_model_and_processor, force_reload),
        asyncio.to_thread(get_emollm_model_and_tokenizer, force_reload),
        asyncio.to_thread(get_tts_model, force_reload),
    )


def get_model_and_processor(
    force_reload: bool = False,
) -> tuple[Qwen2AudioForConditionalGeneration, AutoProcessor, torch.device]:
    """
    兼容旧接口，返回音频理解模型、处理器与输入设备。 

    参数：
        force_reload: 是否强制重载音频模型相关缓存。

    返回：
        与 `get_audio_model_and_processor` 一致的三元组。
    """
    return get_audio_model_and_processor(force_reload=force_reload)



def get_model() -> Qwen2AudioForConditionalGeneration:
    """
    获取音频理解模型实例。 

    参数：
        无。

    返回：
        返回已缓存的 Qwen2-Audio 模型实例。
    """
    model, _, _ = get_audio_model_and_processor()
    return model


def get_processor() -> AutoProcessor:
    """
    获取音频理解处理器实例。 

    参数：
        无。

    返回：
        返回已缓存的 Qwen2-Audio 处理器实例。
    """
    _, processor, _ = get_audio_model_and_processor()
    return processor


def get_input_device() -> torch.device:
    """
    获取音频理解模型的输入设备。 

    参数：
        无。

    返回：
        返回用于音频理解推理输入的 `torch.device`。
    """
    _, _, input_device = get_audio_model_and_processor()
    return input_device


__all__ = [
    "get_model",
    "get_processor",
    "get_input_device",
    "get_model_and_processor",
    "get_audio_model_and_processor",
    "get_emollm_model_and_tokenizer",
    "get_tts_model",
    "warmup_all_models",
    "release_audio_model_and_processor",
    "release_emollm_model_and_tokenizer",
    "release_tts_model",
    "release_all_models",
]
