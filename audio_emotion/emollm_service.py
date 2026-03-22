from __future__ import annotations

import torch

from .load import get_emollm_model_and_tokenizer


DEFAULT_COMPANION_SYSTEM_PROMPT = (
    "你是一位面向老年用户的心理健康与情感支持陪护助手。"
    "你的回复必须做到可陪伴、可引导、可持续："
    "先共情，再给出简单可执行建议，最后给出温和的后续交流引导。"
    "请使用中文，语气温暖、尊重、不说教，不做医疗诊断。"
)


def generate_companion_reply(
    transcription: str,
    emotion: str,
    max_new_tokens: int = 256,
) -> str:
    """
    基于用户转写与情绪生成面向老人的陪护回复文本。

    参数：
        transcription: 用户本轮语音的中文转写文本。
        emotion: 用户本轮整体情绪标签。
        max_new_tokens: 回复生成的最大 token 数，避免过长输出。

    返回：
        返回模型生成的中文回复文本，内容应包含共情、引导和持续对话建议。
    """
    model, tokenizer, input_device = get_emollm_model_and_tokenizer()

    user_prompt = (
        "请根据以下用户输入生成一段老人陪护场景下的回复。\n"
        f"用户转写：{transcription}\n"
        f"用户情绪：{emotion}\n"
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

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(text, return_tensors="pt")
    model_inputs = {
        key: value.to(input_device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    prompt_len = model_inputs["input_ids"].shape[1]
    output_ids = output_ids[:, prompt_len:]
    reply = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return reply or "我在这儿陪着您，愿意和我再慢慢说说刚刚让您在意的事吗？"
