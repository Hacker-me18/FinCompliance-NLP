# -*- coding: utf-8 -*-
"""
数据集 Schema 定义（文档第 7.3 节 - 数据集格式规范）

标准训练数据集 contract_dataset.csv 字段固定、不可增减：

| 字段名     | 字段说明                                       | 字段类型   |
|------------|-----------------------------------------------|-----------|
| id         | 样本唯一 ID，自增编码                           | 字符串/数字 |
| text       | 单条合同条款纯文本                              | 字符串     |
| label      | 风险标签 ID（0-5）                             | 整型       |
| source     | 数据来源：public公开/law司法/synthetic合成      | 字符串     |
| confidence | 标签标注可信度（0-1）                           | 浮点型     |

单条样本示例：
  001,乙方交货完成180日后付款,1,synthetic,0.85
"""

# 标准 Schema 字段列表（冻结不可修改）
DATASET_SCHEMA = ["id", "text", "label", "source", "confidence"]


def validate_schema(df) -> bool:
    """
    验证 DataFrame 是否符合标准 Schema。

    Args:
        df: 待验证的 DataFrame

    Returns:
        bool: 是否符合标准 Schema

    Raises:
        ValueError: 字段缺失或多余时抛出
    """
    if set(df.columns) != set(DATASET_SCHEMA):
        raise ValueError("字段缺失或多余")
    # # 待后续迭代完善
    return True


    raise NotImplementedError("Schema 校验待实现")
