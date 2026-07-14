# -*- coding: utf-8 -*-
"""
数据集划分模块（文档第 12 节 Day2 - 数据集合并划分）

划分策略：
  - 训练集:验证集:测试集 = 8:1:1
  - 分层抽样（stratify）：保证 6 类标签在三个集合中分布一致
  - 输出 train.csv / val.csv / test.csv

输入：contract_dataset.csv（全量合并数据集）
输出：train.csv / val.csv / test.csv
"""


def split_dataset(dataset_path: str, output_dir: str,
                  train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    对全量数据集执行分层抽样划分。

    Args:
        dataset_path: 全量数据集路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    raise NotImplementedError("数据集划分待实现")
