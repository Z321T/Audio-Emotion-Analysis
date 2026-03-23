# 测试情感大模型回复

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

from audio_emotion import (
    load_emollm_model,
    emollm_reply,
)

load_dotenv()  # 从 .env 文件加载环境变量

if __name__ == "__main__":

    # test_input = {
    #     'asr': '烟花在天空中留下的光点消失之后会变成雨会变成雪重新落到地上。他们滋润了大地，抚育了人类，为的是再一次被送上天空，重新华丽的瞬间。在老爹讲的很多很多故事里，这是我很喜欢的一个。',
    #     'emotion': '感慨'
    # }

    test_input = {
        'asr': '我觉得自己好像没有什么朋友，感觉很孤独。',
        'emotion': '孤独'
    }

    # 加载模型
    model_dir = Path(os.getenv("MODEL_PATH_AJUPYTER_EMOLLM_QWEN2_7B_INSTRUCT_LORA"))
    emollm_model, emollm_tokenizer = load_emollm_model(model_dir)

    # 测试情感大模型回复
    result = asyncio.run(
        emollm_reply(
            asr_input=test_input,
            emollm_model=emollm_model,
            emollm_tokenizer=emollm_tokenizer
        )
    )

    print(result)