import sys

from audio_emotion import analyze_audio_file, interactive_microphone_dialog


def main() -> None:
    if len(sys.argv) < 2:
        print("用法:")
        print("  python main.py <本地音频路径>")
        print("  python main.py mic")
        return

    arg = sys.argv[1].strip().lower()
    if arg == "mic":
        interactive_microphone_dialog(max_seconds_per_turn=30)
        return

    response = analyze_audio_file(sys.argv[1])
    print(response)


if __name__ == "__main__":
    main()