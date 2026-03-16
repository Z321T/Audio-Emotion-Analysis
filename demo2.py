from audio_emotion import interactive_microphone_dialog


def main() -> None:
    print("Demo2: 麦克风轮次对话示例")
    print("说明: 每轮语音最长 30 秒，输入 q 可结束。")
    interactive_microphone_dialog(max_seconds_per_turn=30, max_new_tokens=256)


if __name__ == "__main__":
    main()