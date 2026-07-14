# -*- coding: utf-8 -*-
"""
ContractGuard-NLP 核心业务源码包

模块组织（对齐文档第 9 节系统架构）：
  - parsing          文件解析层（PDF/DOCX/TXT 文本提取 + 分句拆分）
  - data_process     数据预处理层（清洗 + 加载 + 划分 + Schema）
  - weak_supervision 弱监督标注层（关键词规则引擎 + 标签生成）
  - llm_synthetic    LLM 合成数据层（专业 Prompt 模板 + 批量生成）
  - train            模型训练层（Stage1 基线 / Stage2 FastText / Stage3 LoRA）
  - evaluate         模型评估层（Recall / F1 / Precision / Accuracy）
  - inference        线上推理层（风险分类推理）
  - llm_enhance      LLM 增强层（风险解释 + RAG 知识库）
  - report           报告生成层（标准化财务风险审核报告）

处理流水线：
  上传合同 → parsing 解析分句 → data_process 清洗标准化
  → inference 风险分类 → llm_enhance 风险解释 → report 报告生成
"""
