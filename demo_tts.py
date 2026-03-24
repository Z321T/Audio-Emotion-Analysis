# 测试TTS模型

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

from audio_emotion import (
    load_qwen_tts_model,
    audio_tts,
)

load_dotenv()  # 从 .env 文件加载环境变量

if __name__ == "__main__":

    test_input = "你好，欢迎使用Qwen TTS模型进行文本到语音的合成测试！希望你喜欢这个温柔的声音。"

    # 加载模型
    model_dir = Path(os.getenv("MODEL_PATH_QWEN_QWEN3_TTS_12HZ_1_7B_CUSTOMVOICE"))
    qwen_tts_model = load_qwen_tts_model(model_dir)


    # 测试Qwen TTS模型
    result_path = asyncio.run(
        audio_tts(
            test_input,
            qwen_tts_model=qwen_tts_model,
        )
    )

    print(f"生成的音频文件路径: {result_path}")