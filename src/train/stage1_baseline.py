# -*- coding: utf-8 -*-
"""
Stage1：基线模型（文档第 8 节 Stage1 - Baseline）

模型：TF-IDF + 随机森林
核心目的：快速验证分类任务可行性，建立最低指标基线，用于后续模型效果对比
技术流程：文本清洗 → TF-IDF 特征提取 → 随机森林分类 → 指标评估
作用：快速兜底，验证数据与任务合理性，不参与线上推理

输出：
  - models/baseline/tfidf_vectorizer.pkl（TF-IDF 向量化器）
  - models/baseline/random_forest.pkl（随机森林分类器）
  - logs/evaluate/baseline_metrics.json（基线指标）
"""

#label1和3的条款较少

import os
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from src.evaluate.metrics import compute_metrics
from src.data_process.load_dataset import load_data_cut
from src.utils.config import get_config


def train_baseline():
    """
    训练 TF-IDF + 随机森林基线模型。
    """
    # 数据加载
    data = load_data_cut()
    words = data["words"]
    labels = data["label"]

    # 先划分再向量化，避免测试集信息泄漏进 vectorizer
    x_train, x_test, y_train, y_test = train_test_split(
        words, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # TF-IDF：只在训练集上 fit，测试集用 transform 转换
    tfidf_vectorizer = TfidfVectorizer(max_features=5000,
                                       ngram_range=(1, 2),
                                       max_df=0.95,
                                       min_df=1)
    x_train_vec = tfidf_vectorizer.fit_transform(x_train)
    x_test_vec = tfidf_vectorizer.transform(x_test)

    # 随机森林模型
    rf = RandomForestClassifier(n_estimators=100,
                                class_weight="balanced",
                                max_features="sqrt",
                                random_state=42)
    rf.fit(x_train_vec, y_train)

    # 评估测试集（不是训练集），才是模型真实表现
    pred = rf.predict(x_test_vec)
    acc, recall, f1, precision, report = compute_metrics(y_test, pred)
    print("准确率-->", acc)
    print("召回率-->", recall)
    print("F1-->", f1)
    print("精确率-->", precision)
    print("分类报告-->", report)

    # 交联验证：多分类必须用 f1_macro
    scores = cross_val_score(rf, tfidf_vectorizer.transform(words), labels, cv=5, scoring="f1_macro")
    print("交叉验证宏 F1-->", scores)
    print("宏 F1 均值-->", scores.mean())

    # 保存模型到项目根目录下的 models/baseline/
    root, cfg = get_config()
    out_dir = root / Path(cfg["train"]["baseline"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    vec_path = out_dir / "tfidf_vectorizer.pkl"
    rf_path = out_dir / "random_forest.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"[保存] 模型已保存到 {out_dir}")


def load_baseline_model(config: dict):
    """
    加载已保存的基线模型（TF-IDF + 随机森林）。
    Returns:
        (tfidf_vectorizer, rf_model)
    """
    root, cfg = get_config()
    out_dir = root / Path(config["train"]["baseline"]["output_dir"])
    with open(out_dir / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(out_dir / "random_forest.pkl", "rb") as f:
        rf_model = pickle.load(f)
    return tfidf_vectorizer, rf_model


if __name__ == "__main__":
    train_baseline()
