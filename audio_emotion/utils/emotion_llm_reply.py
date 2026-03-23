# 情感分析回复处理

from __future__ import annotations

import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM



DEFAULT_COMPANION_SYSTEM_PROMPT = (
    "你是一位面向老年用户的心理健康与情感支持陪护助手。"
    "你的回复必须做到可陪伴、可引导、可持续："
    "先共情，再给出简单可执行建议，最后给出温和的后续交流引导。"
    "请使用中文，语气温暖、尊重、不说教，不做医疗诊断。"
)



def _resolve_input_device(model: AutoModelForCausalLM) -> torch.device:
    """
    根据模型的 device_map 自动推断输入数据应该放在哪个设备上。
    优先使用模型的 hf_device_map 中的设备（如果存在且有效），否则根据当前环境自动选择 CUDA 或 CPU。
    """
    if hasattr(model, "hf_device_map"):
        for mapped_device in model.hf_device_map.values():
            if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk", "meta"}:
                return torch.device(mapped_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



async def emollm_reply(
    asr_input: dict[str, str],
    *args,
    emollm_model: torch.nn.Module,
    emollm_tokenizer: AutoTokenizer,
    **kwargs,
) -> str:
    """
    使用情感分析模型生成回复文本。

    参数:
        asr_input: 包含ASR结果的字典，
        *args: 其他位置参数，传递给模型生成函数
        emollm_model: 已加载的情感分析模型
        emollm_tokenizer: 已加载的情感分析分词器
        **kwargs: 其他关键字参数，传递给模型生成函数

    返回:
        模型生成的回复文本
    """
    # 验证输入格式
    if "asr" not in asr_input or "emotion" not in asr_input:
        raise ValueError("asr_input必须包含'asr'和'emotion'键")
    

    user_prompt = (
        "请根据以下用户输入生成一段老人陪护场景下的回复。\n"
        f"用户转写：{asr_input['asr']}\n"
        f"用户情绪：{asr_input['emotion']}\n"
        "要求：\n"
        "1. 先表达理解与陪伴感。\n"
        "2. 给出一条简单可执行的情绪调节建议。\n"
        "3. 最后给出一个温和追问，鼓励用户继续交流。\n"
        "4. 回复简洁自然，控制在 80 字以内。"
    )

    messages = [
        {"role": "system", "content": DEFAULT_COMPANION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    model, tokenizer = emollm_model, emollm_tokenizer

    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    input_device = _resolve_input_device(model)
    model_inputs = {
        key: value.to(input_device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, max_new_tokens=256)

    prompt_len = model_inputs["input_ids"].shape[1]
    output_ids = output_ids[:, prompt_len:]
    reply = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return reply or "我在这儿陪着您，愿意和我再慢慢说说刚刚让您在意的事吗？"


        