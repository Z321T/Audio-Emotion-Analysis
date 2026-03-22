import sys
import asyncio

from audio_emotion import analyze_audio_file, run_voice_companion_dialog


def main() -> None:
    """
    项目命令行入口，支持文件分析与语音陪护两种模式。 

    参数：
        无。参数通过命令行传入：
        - `python main.py <本地音频路径>`: 对指定音频执行转写与情绪识别
        - `python main.py mic`: 启动自动语音陪护循环

    返回：
        无。结果通过标准输出打印；语音模式下会同时播放合成语音。
    """
    if len(sys.argv) < 2:
        print("用法:")
        print("  python main.py <本地音频路径>")
        print("  python main.py mic")
        return

    arg = sys.argv[1].strip().lower()
    if arg == "mic":
        asyncio.run(
            run_voice_companion_dialog(
                max_seconds_per_turn=30,
                max_audio_tokens=256,
                max_reply_tokens=256,
                max_turns=None,
            )
        )
        return

    response = analyze_audio_file(sys.argv[1])
    print(response)


if __name__ == "__main__":
    main()