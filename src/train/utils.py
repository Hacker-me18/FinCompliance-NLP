# -*- coding: utf-8 -*-
"""
训练公共工具（文档第 8 节 - 模型迭代技术方案）

各阶段训练共用的工具函数：
  - 随机种子固定（Python / NumPy / PyTorch）
  - 设备检测（auto / cuda / cpu）
  - 统一格式日志（控制台 + 文件双输出）
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int):
    """固定全局随机种子，保证训练可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_config: str = "auto") -> torch.device:
    """
    自动检测并返回训练设备。

    Args:
        device_config: auto 自动检测 / cuda 强制 GPU / cpu 强制 CPU

    Returns:
        torch.device
    """
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def setup_logger(name: str, log_dir: str) -> logging.Logger:
    """
    配置统一格式的训练日志器（控制台 + 文件双输出）。

    Args:
        name: 日志器名称
        log_dir: 日志输出目录

    Returns:
        logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件
    log_file = os.path.join(log_dir, f"{name}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
