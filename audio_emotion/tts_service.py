from __future__ import annotations

import numpy as np

from .load import get_tts_model


def synthesize_reply_audio(
    reply_text: str,
    ref_audio: tuple[np.ndarray, int],
) -> tuple[np.ndarray, int]:
    """
    使用 Qwen3-TTS 将陪护回复文本合成为语音波形。

    参数：
        reply_text: 需要播报的中文回复文本。
        ref_audio: 参考语音 `(waveform, sample_rate)`，用于提取说话人特征。

    返回：
        返回 `(wav, sample_rate)`：
        - `wav`: 合成后的单段语音波形
        - `sample_rate`: 对应采样率
        若合成结果为空会直接抛出异常。
    """
    tts_model = get_tts_model()
    wavs, sample_rate = tts_model.generate_voice_clone(
        text=reply_text,
        language="Chinese",
        ref_audio=ref_audio,
        x_vector_only_mode=True,
        max_new_tokens=1024,
    )
    if not wavs:
        raise RuntimeError("TTS 未返回有效语音结果")
    return wavs[0], sample_rate


def play_audio_blocking(audio_waveform: np.ndarray, sample_rate: int) -> None:
    """
    在本地扬声器同步播放语音波形。

    参数：
        audio_waveform: 待播放的一维音频波形。
        sample_rate: 波形采样率。

    返回：
        无。若播放设备不可用会直接抛出异常。
    """
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("语音播放需要安装 sounddevice，先执行: pip install sounddevice") from exc

    sd.play(audio_waveform, samplerate=sample_rate)
    sd.wait()
