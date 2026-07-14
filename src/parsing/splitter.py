# -*- coding: utf-8 -*-
"""
条款级分句拆分模块

职责：
  将合同全文按中文标点规则切分为单条独立条款，为风险识别提供标准化输入。

分句规则：
  - 按中文句号（。）、分号（；）、换行等切分
  - 过滤空行与过短片段（长度 < 阈值则丢弃）
  - 保留条款语义完整性（避免在数字/百分比中间切断）

输入：合同全文文本
输出：单条独立条款文本列表
"""

# 默认分句正则模式（可按需调整）
DEFAULT_SPLIT_PATTERN = r"[。；;\n]+"

# 条款最小长度阈值（字符数），低于此值视为无效片段
MIN_CLAUSE_LENGTH = 5


def split_into_clauses(text: str, pattern: str = None, min_length: int = None) -> list:
    """
    将合同全文切分为独立条款列表。

    Args:
        text: 合同全文文本
        pattern: 分句正则模式（默认按 。；; \n 切分）
        min_length: 条款最小长度阈值

    Returns:
        list[str]: 清洗后的独立条款文本列表
    """
    import re
    pattern = pattern or DEFAULT_SPLIT_PATTERN
    min_length = min_length or MIN_CLAUSE_LENGTH

    # # 待后续迭代完善
    # raw_clauses = re.split(pattern, text)
    # clauses = [c.strip() for c in raw_clauses if len(c.strip()) >= min_length]
    # return clauses
    raise NotImplementedError("分句拆分待实现")
