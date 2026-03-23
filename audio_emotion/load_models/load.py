from __future__ import annotations
from pathlib import Path

import torch
from modelscope import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor, 
    AutoTokenizer,
    AutoModelForCausalLM,
)


# AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
# EMOLLM_MODEL_ID = "aJupyter/EmoLLM_Qwen2-7B-Instruct_lora"
# QWEN_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"



def load_audio_model(
    model_path: Path, 
    *args,
    **kwargs,
) -> tuple[torch.nn.Module, AutoProcessor]:
    """
    加载音频分析模型

    参数:
        model_path: 模型本地路径，直接加载
        *args: 其他位置参数，传递给模型加载函数
        model_id: 模型ID，默认为 "Qwen/Qwen2-Audio-7B-Instruct"
        **kwargs: 其他关键字参数，传递给模型加载函数
    返回:
        加载好的模型对象、处理器对象
    """
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto",
        # device_map="cuda:0",
        )

    return model, processor


def load_emollm_model(
    model_path: Path, 
    *args,
    **kwargs,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """
    加载情感分析模型，若本地不存在则自动下载。

    参数:
        model_path: 模型本地路径，直接加载
        *args: 其他位置参数，传递给模型加载函数
        **kwargs: 其他关键字参数，传递给模型加载函数
    返回:
        加载好的模型对象、分词器对象
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        # device_map="auto",
        device_map="cuda:0",
        )

    return model, tokenizer
