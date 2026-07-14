# -*- coding: utf-8 -*-
"""
多源数据加载与合并模块（文档第 7.1 节 - 数据来源方案）

数据来源：
  1. public 公开：公开中文采购/销售/招投标合同范本（10000 条）
  2. law 司法：公开法院合同财务纠纷文书（5000 条）
  3. synthetic 合成：LLM 生成高仿真风险合同条款（20000 条）

职责：
  - 分别加载三种来源的原始数据
  - 统一字段格式，合并输出 contract_dataset.csv
  - 校验 Schema 合规性
"""
import jieba
import pandas as pd
from src.utils.config import get_config

#读取配置文件
root, config = get_config()
csv_path = root  / config["data"]["dataset_file"]

def load_data_cut():
    """
    加载三源数据 → 统一 Schema → 合并输出。

    Args:
        public_path: 公开合同范本数据路径
        law_path: 司法裁判案例数据路径
        synthetic_path: LLM 合成数据路径

    """
    data = pd.read_csv(csv_path, header=0, encoding="utf-8")
    data = data.loc[:, ["text", "label"]]
    data["words"] = data["text"].apply(lambda x: " ".join(jieba.lcut(x)))
    return data






if __name__ == "__main__":
    pass






