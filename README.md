# Audio-Emotion-Analysis

基于 Qwen2-Audio 的音频理解项目，用于完成以下任务：

- 音频转写（中文）
- 情绪识别（输出整体情绪与简要理由）
- 本地文件批量测试
- 麦克风轮次对话（伪流式：按轮录音后推理）

模型输出统一为 JSON 字符串，格式如下：

```json
{"transcription":"...","emotion":"...","reason":"..."}
```

## 项目特点

- 模型加载解耦：模型与处理器加载在 `audio_emotion/load.py`，可复用单例实例，避免重复加载。
- 推理接口清晰：`audio_emotion/inference.py` 同时支持文件推理与麦克风轮次对话。
- 中文输出约束：系统提示词固定要求中文回答。
- 安全时长限制：麦克风单轮最长 30 秒，超时自动停止录音并开始推理。

## 项目结构

```text
audio_emotion/
	load.py                 # 模型与处理器加载（单例缓存）
	inference.py            # 推理接口（文件/麦克风）
	load_models/
		download_model.py     # 模型下载

demo1.py                  # 单音频/多音频文件测试示例
demo2.py                  # 麦克风轮次对话示例
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

# 麦克风轮次对话
uv run python main.py mic
```

### 2) Demo1：单音频/多音频文件测试

```powershell
uv run python demo1.py
```

可在 `demo1.py` 中替换本地音频路径。

### 3) Demo2：轮次对话接口示例

```powershell
uv run python demo2.py
```

运行后可持续多轮：

- 回车开始一轮录音
- 单轮最长 30 秒
- 输入 `q` 退出

## 核心接口说明

`audio_emotion/inference.py` 提供：

- `analyze_audio_file(audio_path, max_new_tokens=256)`
	- 输入本地音频路径
	- 返回 JSON 字符串（转写 + 情绪 + 理由）

- `interactive_microphone_dialog(max_seconds_per_turn=30, max_new_tokens=256)`
	- 麦克风轮次对话
	- 手动退出，支持连续多轮

`audio_emotion/load.py` 提供：

- `get_model_and_processor(force_reload=False)`
- `get_model()`
- `get_processor()`
- `get_input_device()`

用于在批量任务中复用模型实例，减少重复初始化开销。

## 注意事项

- 麦克风功能依赖 `sounddevice`，请确认系统录音设备可用。
- 首次运行会下载并加载模型，耗时较长属正常现象。
- 若显存不足，`device_map="auto"` 会自动在 CPU/GPU 间分配参数。

