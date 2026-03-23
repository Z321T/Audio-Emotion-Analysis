# 手动下载模型，仅需要执行一次
from audio_emotion import download_model

if __name__ == "__main__":

    # path = download_model("Qwen/Qwen2-Audio-7B-Instruct")
    path = download_model("aJupyter/EmoLLM_Qwen2-7B-Instruct_lora")
    # path = download_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    print(path)