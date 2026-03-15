from typing import Optional
from pathlib import Path

from modelscope import snapshot_download


def download_model(
    model_id: str = "Qwen/Qwen2-Audio-7B-Instruct",
    local_dir: Optional[str] = None,
) -> Path:
    """
    下载/缓存模型到本地目录；已存在则直接复用。
    下载目录固定为与当前文件所在的 `load_models` 目录同级的 `pretrained_models` 下。

    参数:
        model_id: 模型ID，默认为 "Qwen/Qwen2-Audio-7B-Instruct"
        local_dir: 本地存储目录，若为 None 则放在 audio_emotion 目录下的 `pretrained_models/<model_id>` 目录中
    返回:
        模型本地路径
    """

    # 找到该文件的祖父目录 -> audio_emotion 目录
    model_dir = Path(__file__).resolve().parent.parent

    # 若未指定 local_dir，则放在 project/pretrained_models/<model_id>
    local_dir_path = Path(local_dir) if local_dir else model_dir / "pretrained_models" / model_id
    local_dir_path.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        model_id=model_id,
        local_dir=str(local_dir_path),
    )

    model_path = Path(model_path).resolve()


    return model_path