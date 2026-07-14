# -*- coding: utf-8 -*-
"""
风险分类推理模块

输入：单条标准化合同条款文本
输出：结构化风险识别结果
  {
    "risk_type": "payment_risk",
    "risk_label_id": 1,
    "confidence": 0.93,
    "risk_level": "高风险"
  }

通过 config["inference"]["model_type"] 切换模型：
  - "baseline" : TF-IDF + 随机森林
  - "lora"     : MacBERT + LoRA
"""

import os
import pickle
from pathlib import Path

import jieba
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.train.stage1_baseline import load_baseline_model
from src.utils.config import get_config

# 标签体系
LABEL_NAMES = {
    0: "normal_clause",
    1: "payment_risk",
    2: "tax_invoice_risk",
    3: "delivery_acceptance_risk",
    4: "liability_penalty_risk",
    5: "financial_compliance_risk",
}

RISK_LEVELS = {
    0: "无风险",
    1: "高风险",
    2: "中风险",
    3: "中风险",
    4: "中高风险",
    5: "高风险",
}

# 全局模型缓存（避免每次请求重新加载）
_model_cache = {}

# 合同条款关键词（用于拒识非合同输入；只要命中 ≥1 个就放行，避免误杀真实条款）
CONTRACT_KEYWORDS = [
    "甲方", "乙方", "合同", "条款", "签订", "签署",
    "支付", "付款", "结算", "金额", "费用", "价款",
    "交付", "验收", "交货", "货物", "标的",
    "违约", "违约金", "赔偿", "责任", "罚款",
    "发票", "税务", "税率", "增值税",
    "保密", "知识产权", "争议", "仲裁", "诉讼",
    "工期", "期限", "日期", "届满", "终止", "解除",
    "质量", "标准", "保证", "担保", "抵押",
    "采购", "供应", "销售", "承租", "承包",
    "双方", "约定", "履行", "执行", "生效",
]


def _is_contract_clause(text: str) -> bool:
    """判断文本是否包含合同条款特征（命中 1 个关键词即放行）"""
    if len(text) < 5:
        return False
    return any(kw in text for kw in CONTRACT_KEYWORDS)


def _get_model(config: dict):
    """
    根据 config["inference"]["model_type"] 加载对应模型，带缓存。
    """
    model_type = config.get("inference", {}).get("model_type", "baseline")

    if model_type in _model_cache:
        return _model_cache[model_type]

    if model_type == "baseline":
        tfidf, rf = load_baseline_model(config)
        _model_cache[model_type] = (tfidf, rf)
        return tfidf, rf

    elif model_type == "lora":
        from peft import PeftModel
        root, _ = get_config()
        lora_path = root / "models" / "macbert_lora"
        base_path = lora_path / "chinese-macbert-base"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_path, num_labels=6,
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
        tokenizer = AutoTokenizer.from_pretrained(base_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        _model_cache[model_type] = (model, tokenizer, device)
        return model, tokenizer, device

    else:
        raise NotImplementedError(f"模型类型 {model_type} 暂未实现（可选: baseline / lora）")


def _predict(texts: list, config: dict) -> list:
    """对文本列表做批量推理，返回结构化结果"""
    model_type = config.get("inference", {}).get("model_type", "baseline")

    results = []
    for text in texts:
        # 非合同输入拒识
        if not _is_contract_clause(text):
            results.append({
                "text": text,
                "risk_type": "非合同条款",
                "risk_label_id": -1,
                "confidence": 0.0,
                "risk_level": "无法判断",
                "confidence_array": [],
                "rejected": True,
                "message": "输入内容不像是合同条款，请输入有效的合同文本后再试",
            })
            continue

        if model_type == "baseline":
            tfidf, rf = _get_model(config)
            words = " ".join(jieba.lcut(text))
            x_vec = tfidf.transform([words])
            pred = int(rf.predict(x_vec)[0])
            probas = rf.predict_proba(x_vec)[0]

        elif model_type == "lora":
            model, tokenizer, device = _get_model(config)
            enc = tokenizer(
                text,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
                probas = F.softmax(out.logits, dim=1).cpu().numpy()[0]
                pred = int(probas.argmax())

        label_id = pred
        confidence = float(probas[label_id])
        results.append({
            "text": text,
            "risk_type": LABEL_NAMES[label_id],
            "risk_label_id": label_id,
            "confidence": round(confidence, 4),
            "risk_level": RISK_LEVELS[label_id],
            "confidence_array": probas.tolist() if isinstance(probas, np.ndarray) else list(probas),
            "rejected": False,
        })
    return results


def predict_text(config: dict, text: str) -> dict:
    """单条条款风险分类推理"""
    return _predict([text], config)[0]


def predict_batch(config: dict, texts: list) -> list:
    """批量条款风险分类推理"""
    return _predict(texts, config)


def predict_file(config: dict, file_path: str) -> list:
    """合同文件级推理：解析 → 分句 → 逐条推理（待实现）"""
    raise NotImplementedError("文件级推理待实现")
