# -*- coding: utf-8 -*-
"""
Stage3：MacBERT + LoRA 轻量化微调（文档第 8 节 Stage3 - 线上主模型）

模型：本地 MacBERT + LoRA 适配器 + 6 类线性分类头
损失：Focal Loss（γ=2.0 + 类频反比 α）
输出：adapter 权重 + 评估指标 JSON
"""

import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from peft import LoraConfig, get_peft_model, TaskType

from src.utils.config import get_config
from src.train.utils import set_seed, get_device
from src.evaluate.metrics import confusion_plot



# Dataset


class ContractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }



# Focal Loss


def focal_loss(logits, targets, gamma, alpha):
    """FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)"""
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_w = (1.0 - p_t) ** gamma
    alpha_t = alpha.gather(0, targets)
    ce = F.cross_entropy(logits, targets, reduction="none")
    return (alpha_t * focal_w * ce).mean()



# 训练 / 评估


def train_epoch(model, loader, optimizer, scheduler, alpha, gamma, device):
    model.train()
    total = 0.0
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        loss = focal_loss(out.logits, batch["labels"].to(device), gamma, alpha)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        preds.extend(out.logits.argmax(1).cpu().numpy())
        labels.extend(batch["labels"].numpy())
    return np.array(preds), np.array(labels)



# 主入口


def train_lora(config=None):
    # 配置
    if config is None:
        root, config = get_config()
    else:
        root = Path(__file__).resolve().parents[2]

    seed = config["train"]["seed"]
    device = get_device(config["train"]["device"])
    set_seed(seed)
    print(f"[info] device={device}  seed={seed}")

    # 数据
    df = pd.read_csv(root / config["data"]["dataset_file"])
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels,
        test_size=config["data"]["val_ratio"] + config["data"]["test_ratio"],
        random_state=seed, stratify=labels,
    )
    print(f"[info] train={len(tr_texts)}  val={len(val_texts)}")

    # Focal Loss α：类频反比 → 归一化
    counts = Counter(tr_labels)
    num_cls = config["model"]["num_labels"]
    alpha = np.array([1.0 / counts[i] for i in range(num_cls)], dtype=np.float32)
    alpha = alpha / alpha.sum() * num_cls
    alpha_t = torch.tensor(alpha, dtype=torch.float32).to(device)
    gamma = config["train"]["lora"].get("gamma", 2.0)
    print(f"[info] alpha={alpha.round(3).tolist()}  gamma={gamma}")

    # Tokenizer + Dataset
    model_path = str(root / config["model"]["pretrained_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_len = config["model"]["max_length"]
    bs = config["train"]["lora"]["batch_size"]

    tr_ds = ContractDataset(tr_texts, tr_labels, tokenizer, max_len)
    val_ds = ContractDataset(val_texts, val_labels, tokenizer, max_len)
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    # 模型（MacBERT + LoRA）
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_cls,
        id2label={i: str(i) for i in range(num_cls)},
        label2id={str(i): i for i in range(num_cls)},
    )
    lora_cfg = LoraConfig(
        r=config["train"]["lora"]["r"],
        lora_alpha=config["train"]["lora"]["lora_alpha"],
        target_modules=config["train"]["lora"]["target_modules"],
        lora_dropout=config["train"]["lora"]["lora_dropout"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.print_trainable_parameters()

    # 优化器 + 调度器
    epochs = config["train"]["lora"]["epochs"]
    lr = config["train"]["lora"]["learning_rate"]
    wd = config["train"]["lora"]["weight_decay"]
    warmup = config["train"]["lora"]["warmup_ratio"]

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = len(tr_loader) * epochs
    warmup_steps = int(total_steps * warmup)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 训练循环 + Early Stopping
    patience = config["train"]["lora"].get("patience", 2)
    out_dir = str(root / config["train"]["lora"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    best_f1, no_improve = 0.0, 0
    print(f"[info] 开始训练 {epochs} 个 epoch...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, tr_loader, optimizer, scheduler, alpha_t, gamma, device)
        preds, truths = evaluate(model, val_loader, device)
        val_f1 = f1_score(truths, preds, average="macro", zero_division=0)
        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f}  val_macroF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            model.save_pretrained(out_dir)
            print(f"  ★ best={best_f1:.4f} → 已保存 {out_dir}")
        else:
            no_improve += 1
            print(f"  未提升 {no_improve}/{patience}")
            if no_improve >= patience:
                print("[stop] early stopping")
                break

    # 最终评估 + 落盘指标
    preds, truths = evaluate(model, val_loader, device)

    # 混淆矩阵可视化
    class_names = ["normal(0)", "payment(1)", "tax(2)", "delivery(3)", "liability(4)", "financial(5)"]
    confusion_plot(truths, preds, class_names=class_names, normalize=True)

    report = classification_report(
        truths, preds,
        target_names=[str(i) for i in range(num_cls)],
        zero_division=0, output_dict=True,
    )
    metrics_path = str(root / "logs" / "evaluate" / "lora_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "macro_f1": report["macro avg"]["f1-score"],
            "accuracy": report["accuracy"],
            "macro_avg": report["macro avg"],
            "per_class": {k: v for k, v in report.items() if k.isdigit()},
        }, f, ensure_ascii=False, indent=2)

    # 打印高风险类别召回率（业务要求 ≥92%）
    recall_1 = report["1"]["recall"]
    recall_5 = report["5"]["recall"]
    print(f"\n[高风险类别召回率] payment(1)={recall_1:.4f}  financial(5)={recall_5:.4f}  (目标≥92%)")

    print(f"\n[完成] macro-F1={report['macro avg']['f1-score']:.4f}  accuracy={report['accuracy']:.4f}")
    print(f"[完成] 指标已写入 {metrics_path}")
    return model


if __name__ == "__main__":
    train_lora()
