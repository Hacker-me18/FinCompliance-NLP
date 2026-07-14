# -*- coding: utf-8 -*-
"""
读取配置文件，返回 (项目根目录, 配置字典)
调用方拿到后自行用 root / cfg["xxx"]["yyy"] 拼接路径
"""
import yaml
from pathlib import Path


def get_config() -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[2]
    p = root / "config" / "config.yaml"

    if not p.exists():
        raise FileNotFoundError(f"未找到配置文件: {p}")

    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    return root, cfg


if __name__ == "__main__":
    root, cfg = get_config()
    print(f"project root: {root}")
    print(f"dataset: {root / cfg['data']['dataset_file']}")
    print(f"lora output: {root / cfg['train']['lora']['output_dir']}")
