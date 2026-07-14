# -*- coding: utf-8 -*-
"""
弱监督规则引擎（文档第 6 节风险标签体系 + 第 7.1 节弱监督规则标注）

规则构成：
  为 6 类风险标签分别维护关键词列表 + 语义模式，
  通过关键词命中组合判定风险类型，并输出置信度。

设计约束：
  - 仅作初标注兜底，最终标签需经人工审核或 LLM 校验
  - 高置信度样本可直接入库，中低置信度样本送人工复核
  - 规则需持续迭代优化（根据模型误判案例反哺规则）

标签体系（冻结）：
  0 normal_clause            正常条款
  1 payment_risk             付款周期风险（高风险）
  2 tax_invoice_risk         税务发票风险（中风险）
  3 delivery_acceptance_risk 交付验收风险（中风险）
  4 liability_penalty_risk   违约责任风险（中高风险）
  5 financial_compliance_risk财务合规风险（高风险）
"""

# 各风险类别关键词库（需根据业务持续扩充）
RISK_KEYWORDS = {
    1: {
        "high": ["180日", "360日", "顺延", "延期付款", "资金周转困难"],
        "medium": ["账期", "付款周期", "分阶段支付", "延期"],
    },
    2: {
        "high": ["无法开具增值税专票", "回款后开票", "先付款后开票"],
        "medium": ["发票类型", "开票时间", "开票顺序"],
    },
    3: {
        "high": ["满足使用需求后", "无量化验收", "无验收标准"],
        "medium": ["验收", "验收流程", "验收标准"],
    },
    4: {
        "high": ["所有直接间接经济损失", "无限责任", "全额损失兜底"],
        "medium": ["违约金", "违约责任", "赔偿责任"],
    },
    5: {
        "high": ["私人账户结算", "个人账户", "现金大额交易"],
        "medium": ["关联交易", "付款流程"],
    },
}


def rule_label(text: str) -> dict:
    """
    基于关键词规则对单条条款进行弱监督标注。

    Args:
        text: 单条合同条款文本

    Returns:
        dict: {"label": int, "confidence": float, "matched_rules": list}
    """
    raise NotImplementedError("规则引擎标注待实现")


def rule_batch_label(texts: list) -> list:
    """
    批量弱监督标注。

    Args:
        texts: 文本列表

    Returns:
        list[dict]: 每条文本的标注结果列表
    """
    raise NotImplementedError("批量规则标注待实现")
