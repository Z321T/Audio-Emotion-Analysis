from audio_emotion.load_models.download_model import download_model

if __name__ == "__main__":
    print("正在下载/缓存模型...")
    # model_path = download_model("aJupyter/EmoLLM_Qwen2-7B-Instruct_lora")
    # model_path = download_model("Qwen/Qwen2-Audio-7B-Instruct")
    model_path = download_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    print(f"模型已准备好，路径: {model_path}")