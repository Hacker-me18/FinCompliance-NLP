# -*- coding: utf-8 -*-
"""
LLM 风险解释模块（文档第 5.4 节 - 大模型风险解释模块）

核心能力：
  基于模型识别的风险类型，结合制造业财务规范，自动生成：
  - 风险成因：为什么该条款存在风险
  - 风险影响：对企业资金/税务/合规的具体影响
  - 整改建议：如何修改条款以规避风险
  - 风险等级：低 / 中 / 高

输入：
  - text: 风险条款原文
  - risk_type: 模型识别的风险类型（payment_risk / tax_invoice_risk 等）

输出：
  {
    "risk_cause": str,       # 风险成因
    "risk_impact": str,      # 风险影响
    "rectification": str,    # 整改建议
    "risk_level": str        # 低/中/高
  }
"""


def explain_risk(config: dict, text: str, risk_type: str = None) -> dict:
    """
    调用 LLM 生成风险成因、影响、整改建议、风险等级。

    Args:
        config: 全局配置（含 LLM API 配置）
        text: 风险条款原文
        risk_type: 风险类型（可选，辅助 LLM 更精准解释）

    Returns:
        dict: {"risk_cause": str, "risk_impact": str, "rectification": str, "risk_level": str}
    """
    raise NotImplementedError("LLM 风险解释待实现")
