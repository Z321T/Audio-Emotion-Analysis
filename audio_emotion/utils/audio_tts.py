# 音频TTS合成

from __future__ import annotations
from pathlib import Path

import torch
import soundfile as sf

from .output_path import unique_output_path



async def audio_tts(
    text_input: str,
    *args,
    qwen_tts_model: torch.nn.Module,
    **kwargs,
) -> Path:
    """
    使用 Qwen TTS 模型将文本合成为音频数据。

    参数:
        text_input: 输入文本字符串
        *args: 其他位置参数，传递给模型生成函数
        qwen_tts_model: 已加载的 Qwen TTS 模型
        **kwargs: 其他关键字参数，传递给模型生成函数
    返回:
        合成的音频数据，保存到 audio_emotion/output 目录内并返回文件路径
    """
    # 验证输入格式
    if not isinstance(text_input, str):
        raise ValueError("text_input必须是字符串类型")
    
    model = qwen_tts_model
    
    wavs, sr = model.generate_custom_voice(
        text=text_input,
        language="Chinese",
        speaker="Vivian",
        instruct="用温柔的语气说", 
    )

    # 生成唯一输出文件路径
    output_path = unique_output_path(prefix="tts_output_", ext=".wav")

    # 保存音频数据到文件
    sf.write(output_path, wavs[0], sr)

    return Path(output_path)





# def _split_text_for_streaming(text: str, max_chars: int = 24) -> List[str]:
#     """
#     将长文本切分为适合低延迟生成的小片段。

#     参数:
#         text: 待合成的完整文本。
#         max_chars: 单个片段的最大字符数，值越小首包通常越快。

#     返回:
#         片段列表。若文本为空白，返回空列表。
#     """
#     normalized = re.sub(r"\s+", "", text or "")
#     if not normalized:
#         return []

#     units = [u for u in re.split(r"(?<=[。！？!?；;，,])", normalized) if u]
#     chunks: List[str] = []
#     buffer = ""

#     for unit in units:
#         if len(buffer) + len(unit) <= max_chars:
#             buffer += unit
#             continue

#         if buffer:
#             chunks.append(buffer)

#         if len(unit) <= max_chars:
#             buffer = unit
#         else:
#             for i in range(0, len(unit), max_chars):
#                 chunks.append(unit[i : i + max_chars])
#             buffer = ""

#     if buffer:
#         chunks.append(buffer)

#     return [c for c in chunks if c]


# def _wav_chunk_to_base64(wav: np.ndarray, sample_rate: int) -> str:
#     """
#     将单个波形片段编码为 Base64 WAV 字符串。

#     参数:
#         wav: 单声道波形数据，期望为 float32 一维数组。
#         sample_rate: 波形采样率。

#     返回:
#         Base64 编码后的 WAV 二进制字符串。
#     """
#     with io.BytesIO() as buf:
#         sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
#         raw = buf.getvalue()
#     return base64.b64encode(raw).decode("utf-8")


# async def audio_tts_stream_base64(
#     text_input: str,
#     *args,
#     qwen_tts_model: torch.nn.Module,
#     language: str = "Chinese",
#     speaker: str = "Vivian",
#     instruct: str = "用温柔的语气说",
#     segment_max_chars: int = 24,
#     **kwargs,
# ) -> AsyncGenerator[Dict[str, Any], None]:
#     """
#     使用 Base64 分片方式流式返回 TTS 结果，并在结束事件中返回最终保存路径。

#     参数:
#         text_input: 输入文本字符串，会被自动分段以降低首包等待时间。
#         *args: 预留位置参数，当前不使用，保留向后兼容。
#         qwen_tts_model: 已加载的 Qwen TTS 模型实例。
#         language: 语言参数，默认 Chinese。
#         speaker: 说话人参数，默认 Vivian。
#         instruct: 语气控制指令。
#         segment_max_chars: 文本切片最大长度。
#         **kwargs: 透传给模型生成函数的其它参数。

#     返回:
#         通过异步生成器持续产出事件字典:
#         {
#             "type": "chunk",
#             "seq": 分片序号(从1开始),
#             "sample_rate": 采样率,
#             "audio_base64": Base64 WAV 分片,
#             "text": 本分片对应文本,
#         }
#         结束时额外产出:
#         {
#             "type": "done",
#             "seq": 结束事件序号,
#             "sample_rate": 采样率,
#             "chunks": 分片数量,
#             "duration_seconds": 完整音频时长(秒),
#             "url": 最终音频文件路径字符串,
#         }
#     """
#     if not isinstance(text_input, str):
#         raise ValueError("text_input必须是字符串类型")

#     model = qwen_tts_model
#     text_chunks = _split_text_for_streaming(text_input, max_chars=segment_max_chars)
#     if not text_chunks:
#         raise ValueError("text_input不能为空")

#     wav_chunks: List[np.ndarray] = []
#     sample_rate: int | None = None

#     for seq, text_chunk in enumerate(text_chunks, start=1):
#         wavs, sr = model.generate_custom_voice(
#             text=text_chunk,
#             language=language,
#             speaker=speaker,
#             instruct=instruct,
#             **kwargs,
#         )
#         wav = np.asarray(wavs[0], dtype=np.float32)
#         wav_chunks.append(wav)
#         sample_rate = int(sr)

#         yield {
#             "type": "chunk",
#             "seq": seq,
#             "sample_rate": sample_rate,
#             "audio_base64": _wav_chunk_to_base64(wav, sample_rate),
#             "text": text_chunk,
#         }

#     full_wav = np.concatenate(wav_chunks, axis=0)
#     output_path = unique_output_path(prefix="tts_output_", ext=".wav")
#     sf.write(output_path, full_wav, sample_rate)

#     yield {
#         "type": "done",
#         "seq": len(wav_chunks) + 1,
#         "sample_rate": sample_rate,
#         "chunks": len(wav_chunks),
#         "duration_seconds": round(float(full_wav.shape[0]) / float(sample_rate), 3),
#         "url": str(Path(output_path)),
#     }
