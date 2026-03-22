# Audio-Emotion-Analysis

基于 Qwen2-Audio + EmoLLM + Qwen3-TTS 的语音陪护链路项目，用于完成以下任务：

- 音频转写（中文）
- 情绪识别（输出整体情绪）
- 基于转写与情绪生成陪护回复（面向老人）
- 将回复文本合成为语音并自动播放
- 本地文件批量测试
- 麦克风语音输入输出闭环（自动录音、自动播报）

模型输出统一为 JSON 字符串，格式如下：

```json
{"transcription":"...","emotion":"..."}
```

## 项目特点

- 模型单例复用：`audio_emotion/load.py` 统一管理 Qwen2-Audio、EmoLLM、Qwen3-TTS 的缓存加载。
- 推理链路完整：`audio_emotion/inference.py` 支持音频分析、回复生成、语音合成与自动播报。
- 中文输出约束：系统提示词固定要求中文回答。
- 安全时长限制：麦克风单轮最长 30 秒，超时或检测到静音后自动停止并开始推理。

## 项目结构

```text
audio_emotion/
	load.py                 # 三类模型加载（单例缓存）
	inference.py            # 推理链路（分析/回复/TTS/麦克风）
	load_models/
		download_model.py     # 模型下载

demo1.py                  # 单音频/多音频文件测试示例
demo2.py                  # 自动语音陪护示例（无额外键盘操作）
demo3_full_voice_pipeline.py  # 预热模型 + 完整语音链路演示
main.py                   # 统一入口示例
```

## 环境与依赖

- Python >= 3.10
- 推荐使用 uv 管理环境
- 关键依赖：transformers、torch、librosa、sounddevice、modelscope

安装依赖（在项目根目录）：

```powershell
uv sync
```

## 使用方式

### 1) 统一入口（推荐）

```powershell
# 本地音频文件推理
uv run python main.py test_audio/测试音频001.mp3

# 麦克风自动语音陪护
uv run python main.py mic
```

### 2) Demo1：单音频/多音频文件测试

```powershell
uv run python demo1.py
```

可在 `demo1.py` 中替换本地音频路径。

### 3) Demo2：自动语音陪护示例

```powershell
uv run python demo2.py
```

运行后可持续多轮：

- 每轮自动开始录音
- 检测到静音后自动结束录音并处理
- 自动播放模型语音回复
- 按 `Ctrl+C` 退出

### 4) Demo3：完整链路演示（含预热）

```powershell
uv run python demo3_full_voice_pipeline.py
```

## 核心接口说明

`audio_emotion/inference.py` 提供：

- `analyze_audio_file(audio_path, max_new_tokens=256)`
	- 输入本地音频路径
	- 返回 JSON 字符串（转写 + 情绪）

- `run_voice_companion_dialog(max_seconds_per_turn=30, max_audio_tokens=256, max_reply_tokens=256, max_turns=None)`
	- 自动语音陪护链路（录音 -> 分析 -> 回复 -> 语音播报）
	- 支持连续多轮

- `interactive_microphone_dialog(max_seconds_per_turn=30, max_new_tokens=256)`
	- 兼容旧接口，内部调用自动语音陪护链路

`audio_emotion/load.py` 提供：

- `get_audio_model_and_processor(force_reload=False)`
- `get_emollm_model_and_tokenizer(force_reload=False)`
- `get_tts_model(force_reload=False)`
- `warmup_all_models(force_reload=False)`
- `get_model_and_processor(force_reload=False)`
- `get_model()`
- `get_processor()`
- `get_input_device()`

用于在批量任务中复用模型实例，减少重复初始化开销。

## 注意事项

- 麦克风功能依赖 `sounddevice`，请确认系统录音设备可用。
- 首次运行会下载并加载模型，耗时较长属正常现象。
- 若显存不足，`device_map="auto"` 会自动在 CPU/GPU 间分配参数。

