# -*- coding: utf-8 -*-
"""
评估指标计算模块（文档第 11 节 - 模型评价指标）

指标优先级：Recall > F1 > Precision > Accuracy

输出内容：
  - 整体指标（macro / weighted）
  - 每个风险类别的单独指标（尤其关注高风险类别）
  - 混淆矩阵可视化
  - 核心风险类别（label=1,5）召回率是否达标（≥92%）
"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, \
    classification_report


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    计算整体评估指标。

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        config: 全局配置

    Returns:
        dict: {"recall": float, "f1": float, "precision": float, "accuracy": float}
    """
    rs = recall_score(y_true, y_pred, average="macro")
    ps = precision_score(y_true, y_pred, average="macro")
    fs = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    report  = classification_report(y_true, y_pred)
    # 顺序与 baseline 调用方一致：(acc, recall, f1, precision, report)
    return acc, rs, fs, ps, report



def compute_per_class_metrics(y_true: list, y_pred: list) -> dict:
    """
    计算每个风险类别的单独指标。

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    Returns:
        dict: {label_id: {"recall": float, "f1": float, "precision": float}}
    """
    raise NotImplementedError("分类别指标计算待实现")


def confusion_plot(y_test, y_pred, class_names=None, normalize=False):
    """
    绘制混淆矩阵
    :param y_test: 真实标签
    :param y_pred: 预测标签
    :param class_names: 类别名称列表，如 ['Class 0', 'Class 1', ...]
    :param normalize: 是否按行归一化（显示百分比，便于观察召回率）
    """
    cm = confusion_matrix(y_test, y_pred)

    # 如果需要百分比，按行归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # 保留两位小数
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'  # 整数格式
        title = 'Confusion Matrix'

    # 如果没有传入类名，自动生成
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(8, 6))
    # 使用 seaborn 的 heatmap，自动处理颜色对比度
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='white')

    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
