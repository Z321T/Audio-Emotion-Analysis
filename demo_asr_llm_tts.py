# 全流程测试

import asyncio
import gc
import os
from dotenv import load_dotenv
from pathlib import Path

import torch

from audio_emotion import (
    load_audio_model, 
    analysis_audio_with_path,
    load_emollm_model,
    emollm_reply,
)

load_dotenv()  # 从 .env 文件加载环境变量

def release_cuda_memory(*objects: object) -> None:
    """
    释放一组对象占用的显存，并执行 Python 与 CUDA 层面的内存回收。

    参数:
        *objects: 需要释放的对象列表，通常传入模型、处理器、分词器等。
            函数会尽量将支持 `.to()` 的对象迁回 CPU，随后触发垃圾回收与 CUDA 缓存清理。

    返回:
        无。函数只执行资源回收动作，不返回业务结果。
    """
    for obj in objects:
        if obj is None:
            continue
        try:
            # 由 accelerate hooks 管理的分片模型不应手动 to("cpu")。
            if hasattr(obj, "to") and not hasattr(obj, "hf_device_map"):
                obj.to("cpu")
        except Exception:
            # 部分对象不支持 to("cpu") 或内部状态不完整，直接跳过即可。
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


if __name__ == "__main__":

    # test_audio = Path("test_audio/测试音频001.mp3")
    test_audio = Path("test_audio/测试音频002.mp3")

    # 加载模型
    model_dir = Path(os.getenv("MODEL_PATH_QWEN_QWEN2_AUDIO_7B_INSTRUCT"))
    audio_model, audio_processor = load_audio_model(model_dir)

    # 测试音频转写和情感分析
    asr_result = asyncio.run(
        analysis_audio_with_path(
            test_audio,
            audio_model=audio_model,
            audio_processor=audio_processor,
        )
    )

    print(asr_result)

    # 第一阶段完成后立即卸载 ASR 相关对象，避免与后续大模型争抢显存。
    release_cuda_memory(audio_model, audio_processor)
    audio_model = None
    audio_processor = None
    if torch.cuda.is_available():
        print(f"显存已清理，当前占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("当前为 CPU 环境，无需清理 CUDA 显存。")

    # 加载情感大模型
    model_dir = Path(os.getenv("MODEL_PATH_AJUPYTER_EMOLLM_QWEN2_7B_INSTRUCT_LORA"))
    emollm_model, emollm_tokenizer = load_emollm_model(model_dir)

    input_for_emollm = asr_result

    empllm_result = asyncio.run(
        emollm_reply(
            asr_input=input_for_emollm,
            emollm_model=emollm_model,
            emollm_tokenizer=emollm_tokenizer
        )
    )

    print(empllm_result)

    # 第二阶段结束后同样回收 EmoLLM 显存，便于后续继续加载 TTS 等模型。
    release_cuda_memory(emollm_model, emollm_tokenizer)
    emollm_model = None
    emollm_tokenizer = None