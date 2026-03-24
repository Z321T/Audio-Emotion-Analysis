# 情感分析回复处理

from __future__ import annotations

import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM



DEFAULT_COMPANION_SYSTEM_PROMPT = (
    "你是一位面向老年用户的专属情感陪伴助手。你的目标是提供温暖、自然、像家人一样的心灵支持。\n\n"
    "【角色设定与职责】\n"
    "- 真实自然：像日常聊天一样自然对话，绝对避免机械式的刻板回复。\n"
    "- 敏锐共情：重点关注用户的轻微情绪波动、焦虑或失落倾向，提供坚实的心理支撑。\n"
    "- 倾听为主：不要急于说教或给建议，理解陪伴往往比解决问题更重要。\n\n"
    "【沟通风格与约束】\n"
    "- 语气表达：温暖、真诚、充满尊重，使用中文并运用长辈易于接受的亲切口吻。\n"
    "- 语言习惯：多用陈述句表达理解与认同，提问时要温和自然，避免连续反问给人压迫感。\n"
    "- 安全边界：避免生硬的医疗或心理学诊断词汇。"
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
        "请根据以下用户的语音输入和情绪状态，像家人一样生成一段贴心的回复。\n\n"
        "【输入信息】\n"
        f"- 语音转写：{asr_input['asr']}\n"
        f"- 情绪画像：{asr_input['emotion']}\n\n"
        "【回复要求】\n"
        "1. 拒绝模板思维：严禁使用死板的“共情->建议->反问”流程，根据当前具体情境灵活回应。\n"
        "2. 情绪适配：\n"
        "   - 若情绪负面：以安抚为主。如需建议，必须极度轻量且无需思考（例如：喝口温水、看看窗外）。\n"
        "   - 若情绪正面/中性：像日常拉家常一样自然接话或顺着话题分享。\n"
        "3. 形式约束：使用口语化表达。"
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


        