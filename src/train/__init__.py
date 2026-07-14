# -*- coding: utf-8 -*-
"""
模型训练层（文档第 8 节 - 模型迭代技术方案）

4 阶段渐进式训练：
  Stage1  baseline       TF-IDF + 随机森林（基线，不参与线上推理）
  Stage2  fasttext       FastText 轻量模型（过渡优化）
  Stage3  macbert_lora   MacBERT/RoBERTa + LoRA（线上主模型）
  Stage4  llm_enhance    LLM 增强（辅助能力，不替代分类模型，见 llm_enhance/ 模块）

设计约束：
  - 基线模型仅用于实验对比，线上生产环境仅使用微调后的深度模型
  - 各阶段训练脚本独立，可单独运行与评估
"""
