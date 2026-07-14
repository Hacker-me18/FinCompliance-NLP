# -*- coding: utf-8 -*-
"""
Stage2：轻量模型（文档第 8 节 Stage2 - FastText）

模型：FastText
核心目的：适配合同专业词汇、长尾风险词汇，兼顾训练速度与推理成本
优势：训练极速、推理低成本、适合低资源快速迭代，作为过渡优化模型

输出：
  - models/fasttext/fasttext_model.bin
  - logs/evaluate/fasttext_metrics.json
"""


def train_fasttext(config: dict):
    """
    训练 FastText 轻量分类模型。

    Args:
        config: 全局配置（含 fasttext 超参、数据路径）
    """
    raise NotImplementedError("Stage2 FastText 训练待实现")
