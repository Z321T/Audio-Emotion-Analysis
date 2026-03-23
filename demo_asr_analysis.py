# 测试模型的音频转写和情感分析功能

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

from audio_emotion import (
    load_audio_model, 
    analysis_audio_with_path,
)

load_dotenv()  # 从 .env 文件加载环境变量

if __name__ == "__main__":

    # test_audio = Path("test_audio/测试音频001.mp3")
    test_audio = Path("test_audio/测试音频002.mp3")

    # 加载模型
    model_dir = Path(os.getenv("MODEL_PATH_QWEN_QWEN2_AUDIO_7B_INSTRUCT"))
    audio_model, audio_processor = load_audio_model(model_dir)

    # 测试音频转写和情感分析
    result = asyncio.run(
        analysis_audio_with_path(
            test_audio,
            audio_model=audio_model,
            audio_processor=audio_processor,
        )
    )

    print(result)