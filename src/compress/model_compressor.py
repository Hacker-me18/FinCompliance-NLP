# -*- coding: utf-8 -*-
"""
模型压缩模块

功能：将训练好的模型压缩为 ONNX + 量化格式，减小体积、加速推理
支持：MacBERT + LoRA 分类头（当前），可扩展其他 BERT 系列模型

使用方式：
  # 命令行压缩
  python -m src.compress.model_compressor --model_dir models/macbert_lora --output_dir models/compressed

  # 代码调用
  from src.compress.model_compressor import compress_model
  compress_model(model_dir="models/macbert_lora", output_dir="models/compressed")
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.config import get_config


def compress_model(
    model_dir: str = None,
    output_dir: str = None,
    model_name: str = "macbert",
    quantize: bool = True,
    export_onnx: bool = True,
):
    """
    压缩模型：动态量化 + ONNX 导出

    Args:
        model_dir: 训练好的模型目录（含 adapter_model.safetensors）
        output_dir: 压缩后模型的保存目录
        model_name: 模型标识名称
        quantize: 是否做动态量化
        export_onnx: 是否导出 ONNX
    """
    root, config = get_config()
    device = torch.device("cpu")  # 压缩在 CPU 上做即可

    if model_dir is None:
        model_dir = str(root / config["train"]["lora"]["output_dir"])
    if output_dir is None:
        output_dir = str(root / "models" / "compressed")

    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[压缩] 加载模型: {model_path}")

    # ── 1. 加载模型 ──────────────────────────────────────
    # 加载 LoRA 适配后的完整模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    num_labels = model.config.num_labels
    print(f"[压缩] 类别数: {num_labels} | 原始参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 2. 动态量化 ──────────────────────────────────────
    if quantize:
        print("[压缩] 执行动态量化 (INT8)...")
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear},  # 只量化线性层，对 BERT 效果最好
            dtype=torch.qint8,
        )

        # 保存量化后的 PyTorch 模型
        pt_path = out_path / f"{model_name}_quantized"
        quantized_model.save_pretrained(pt_path)
        tokenizer.save_pretrained(pt_path)
        print(f"[压缩] 量化模型已保存: {pt_path}")

        # 计算压缩率
        orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
        quant_size = sum(
            p.numel() * p.element_size()
            for p in quantized_model.parameters()
            if not p.is_quantized
        ) + sum(
            p.numel() * 1  # qint8 = 1 byte
            for p in quantized_model.parameters()
            if p.is_quantized
        )
        ratio = orig_size / max(quant_size, 1)
        print(f"[压缩] 量化比: {orig_size/1024/1024:.1f}MB → {quant_size/1024/1024:.1f}MB (压缩 {ratio:.1f}x)")

        compress_model = quantized_model
    else:
        compress_model = model
        pt_path = out_path / f"{model_name}"
        model.save_pretrained(pt_path)
        tokenizer.save_pretrained(pt_path)

    # ── 3. ONNX 导出 ────────────────────────────────────
    if export_onnx:
        print("[压缩] 导出 ONNX 格式...")
        onnx_path = out_path / f"{model_name}.onnx"

        # 构造虚拟输入
        dummy_input = tokenizer(
            "这是一条测试合同条款",
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = dummy_input["input_ids"]
        attention_mask = dummy_input["attention_mask"]

        torch.onnx.export(
            compress_model,
            (input_ids, attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )
        onnx_size = onnx_path.stat().st_size / 1024 / 1024
        print(f"[压缩] ONNX 模型已保存: {onnx_path} ({onnx_size:.1f}MB)")

    # ── 4. 保存压缩元信息 ────────────────────────────────
    meta = {
        "model_name": model_name,
        "num_labels": num_labels,
        "compressed": quantize,
        "onnx_exported": export_onnx,
        "compression_method": "dynamic_quantization_int8",
        "labels": model.config.id2label if hasattr(model.config, "id2label") else {},
    }
    meta_path = out_path / f"{model_name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[压缩] 元信息已保存: {meta_path}")
    print("[压缩] 完成！")


def load_compressed_model(model_dir: str):
    """
    加载压缩后的量化模型用于推理（支持 ONNX Runtime 或 PyTorch）

    Args:
        model_dir: 压缩模型目录

    Returns:
        model, tokenizer
    """
    model_path = Path(model_dir)

    # 优先尝试 ONNX Runtime
    onnx_path = model_path.with_suffix(".onnx")
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path))
            tokenizer = AutoTokenizer.from_pretrained(model_path.parent)
            print(f"[加载] ONNX Runtime: {onnx_path}")
            return session, tokenizer, "onnx"
        except ImportError:
            print("[加载] onnxruntime 未安装，回退到 PyTorch")

    # 回退 PyTorch 量化模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, "pytorch"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型压缩工具")
    parser.add_argument("--model_dir", type=str, default=None, help="训练好的模型目录")
    parser.add_argument("--output_dir", type=str, default=None, help="压缩输出目录")
    parser.add_argument("--model_name", type=str, default="macbert", help="模型名称")
    parser.add_argument("--no_quantize", action="store_true", help="不做量化")
    parser.add_argument("--no_onnx", action="store_true", help="不导出 ONNX")
    args = parser.parse_args()

    compress_model(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        quantize=not args.no_quantize,
        export_onnx=not args.no_onnx,
    )
