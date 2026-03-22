import asyncio

from audio_emotion import run_voice_companion_dialog


async def _main() -> None:
    """
    演示完整语音链路：语音输入 -> 转写与情绪 -> 陪护回复 -> TTS播报。 

    参数：
        无。

    返回：
        无。启动后按步骤懒加载模型并在每步后释放；按 `Ctrl+C` 结束程序。
    """
    print("Demo3: 完整语音陪护链路演示")
    print("已启用低显存模式：按步骤加载模型，用完即释放。")

    await run_voice_companion_dialog(
        max_seconds_per_turn=30,
        max_audio_tokens=256,
        max_reply_tokens=256,
        max_turns=None,
    )


def main() -> None:
    """
    Demo 启动入口。 

    参数：
        无。

    返回：
        无。内部通过 `asyncio.run` 启动完整异步语音流程。
    """
    asyncio.run(_main())


if __name__ == "__main__":
    main()
