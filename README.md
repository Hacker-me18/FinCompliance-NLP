# FinCompliance-NLP

金融合规对话意图识别系统，基于 BERT 微调实现金融领域对话的意图分类与合规检测。

## 项目结构

```
FinCompliance-NLP/
├── .gitignore                  # Git 文件过滤规则
├── README.md                   # 项目文档
├── requirements.txt            # 项目依赖清单
├── main.py                     # 程序统一入口
├── config/                     # 全局配置目录
│   └── config.yaml             # 数据、模型、训练、压缩超参配置
├── data/                       # 数据集目录（不上传 Git）
│   ├── raw/                    # 原始金融对话数据集
│   ├── processed/              # 清洗划分后的训练 / 验证 / 测试集
│   └── synthetic/              # LLM 生成扩充仿真样本
├── src/                        # 核心业务源码
│   ├── data_process/           # 数据集处理模块
│   │   ├── load_dataset.py     # 加载原始数据集
│   │   ├── clean_data.py       # 数据清洗
│   │   └── split_train_val.py  # 训练 / 验证 / 测试集划分
│   ├── train/                  # 模型微调训练脚本
│   │   ├── base_bert_train.py  # BERT 基座模型微调
│   │   └── lora_finetune.py    # LoRA 参数高效微调
│   ├── evaluate/               # 训练指标评估
│   │   └── metric.py           # accuracy / F1 / precision / recall
│   ├── inference/              # 线上推理预测模块
│   │   └── predict.py          # 推理预测
│   ├── model_compress/         # 模型压缩方案
│   │   ├── quantize.py         # 模型量化
│   │   ├── distill.py          # 知识蒸馏
│   │   └── prune.py            # 模型剪枝
│   └── generate_aug_data.py    # 金融样本数据增强生成
├── models/                     # 模型权重（不上传 Git）
│   ├── bert_base/              # BERT 基座微调权重
│   ├── lora_adapter/           # LoRA 适配器权重
│   ├── quantized_model/        # 量化轻量化模型
│   └── distilled_student/      # 知识蒸馏学生模型
├── logs/                       # 训练、压缩实验日志（不上传 Git）
├── assets/                     # 实验图表、截图、文档素材
└── demo/                       # 模型部署演示服务
    └── api.py                  # FastAPI 推理服务
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 查看帮助
python main.py --help

# 数据处理
python main.py --task process

# 模型训练
python main.py --task train

# 模型评估
python main.py --task evaluate

# 推理预测
python main.py --task predict

# 模型压缩
python main.py --task compress
```

## 配置说明

所有超参数集中在 `config/config.yaml`，包括数据路径、模型选择、训练参数、压缩方案等。

## 当前状态

| 模块 | 状态 |
|------|------|
| 项目骨架 | ✅ 已搭建 |
| 数据处理 | 🔲 待开发 |
| 模型训练 | 🔲 待开发 |
| 模型评估 | 🔲 待开发 |
| 推理预测 | 🔲 待开发 |
| 模型压缩 | 🔲 待开发 |
| 部署演示 | 🔲 待开发 |
