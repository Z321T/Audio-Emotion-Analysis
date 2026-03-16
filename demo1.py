from audio_emotion import analyze_audio_file


def main() -> None:

    audio_path = "test_audio/测试音频001.mp3"
    response = analyze_audio_file(audio_path)

    print(response)

    audio_path = "test_audio/测试音频002.mp3"
    response = analyze_audio_file(audio_path)
    print(response)


if __name__ == "__main__":
    main()