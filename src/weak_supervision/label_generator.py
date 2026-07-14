# -*- coding: utf-8 -*-
"""
弱监督标签生成器（文档第 7.1 节 - 弱监督规则标注）

职责：
  整合规则引擎输出，对全量无标签数据生成伪标签，
  按置信度分档：高置信度自动入库，中置信度人工复核，低置信度丢弃。

置信度分档：
  - high   (≥0.85): 自动入库
  - medium (0.6-0.85): 人工复核
  - low    (<0.6): 丢弃或送人工标注
"""


def generate_weak_labels(texts: list, config: dict) -> list:
    """
    对文本列表批量生成弱监督标签。

    Args:
        texts: 待标注文本列表
        config: 全局配置（含关键词规则、置信度阈值）

    Returns:
        list[dict]: 标注结果列表，每条包含 label / confidence / source=weak
    """
    raise NotImplementedError("弱监督标签生成待实现")
