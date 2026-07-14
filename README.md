# ContractGuard-NLP

> 基于 NLP 的**制造业合同财务风险智能识别系统**
>
> 自动识别制造业合同条款中的 6 类财务风险，支持本地 Web 演示与 API 服务调用。

---

## 项目简介

制造业合同条款内嵌大量核心财务约束条件（付款周期、发票规范、验收标准、违约责任、结算方式等），直接影响企业现金流、税务合规与审计合规。

本项目依托 NLP 深度学习技术，实现：**合同文本输入 → 风险分类 → 置信度输出** 的全流程自动化识别。

---

## 风险标签体系

| 标签 ID | 风险类型 | 说明 | 风险等级 |
|---------|----------|------|----------|
| 0 | `normal_clause` 正常条款 | 无财务风险的常规履约条款 | 无风险 |
| 1 | `payment_risk` 付款周期风险 | 账期过长、超高预付款、分期回款不合理 | **高风险** |
| 2 | `tax_invoice_risk` 税务发票风险 | 开票延迟、无法开具专票、发票类型模糊 | 中风险 |
| 3 | `delivery_acceptance_risk` 交付验收风险 | 验收标准模糊、无量化指标、无明确时限 | 中风险 |
| 4 | `liability_penalty_risk` 违约责任风险 | 责任不对等、高额违约金、无限责任 | 中高风险 |
| 5 | `financial_compliance_risk` 财务合规风险 | 现金大额交易、个人账户结算 | **高风险** |

---

## 系统架构

```
合同文本输入
    ↓
文本预处理（清洗 + 分句）
    ↓
风险分类模型（TF-IDF + RF / MacBERT + LoRA 可选）
    ↓
结构化结果输出（风险类型 + 置信度 + 概率分布）
    ↓
Web 前端展示 / API 服务
```

---

## 模型方案

| 阶段 | 模型 | 定位 | 说明 |
|------|------|------|------|
| Stage1 | TF-IDF + 随机森林 | 基线模型 | 快速验证，关键词特征明显时效果好 |
| Stage2 | FastText | 轻量过渡模型 | 适配合同专业词汇（待完善） |
| Stage3 | MacBERT + **LoRA** | **线上主模型** | 语义理解强，解决嵌套隐性风险识别 |

通过 `config/config.yaml` 中的 `inference.model_type` 一键切换：

```yaml
inference:
  model_type: baseline   # baseline / lora
```

---

## 项目结构

```
ContractGuard-NLP/
├── main.py                  # 程序统一入口（命令行任务调度）
├── README.md                # 项目文档
├── requirements.txt         # 依赖清单
├── config/
│   └── config.yaml          # 全局超参配置
│
├── src/                     # 核心源码
│   ├── train/               # 模型训练（baseline / lora）
│   ├── inference/           # 推理逻辑（支持 baseline / lora 切换）
│   ├── compress/            # 模型压缩（动态量化 + ONNX 导出）
│   ├── evaluate/            # 评估指标 + 混淆矩阵
│   ├── data_process/        # 数据加载与预处理
│   ├── parsing/             # 文件解析（PDF/DOCX/TXT）
│   ├── weak_supervision/    # 弱监督标注
│   ├── llm_synthetic/       # LLM 合成数据
│   ├── llm_enhance/         # LLM 风险解释
│   ├── report/              # 报告生成
│   └── utils/               # 公共工具
│
├── demo/                    # Web 前端 + API 服务
│   ├── api.py               # FastAPI 推理接口
│   ├── schemas.py           # 请求/响应数据模型
│   └── templates/
│       └── index.html       # Vue.js 前端演示页面
│
├── models/                  # 模型权重（不上传 Git）
│   ├── baseline/            # TF-IDF + RF 模型
│   ├── macbert_lora/        # MacBERT + LoRA 模型
│   └── compressed/          # 压缩后模型
│
├── data/                    # 数据集（不上传 Git）
│   ├── raw/                 # 原始数据
│   ├── processed/           # 清洗后数据集
│   └── synthetic/           # LLM 合成数据
│
└── logs/                    # 训练/评估日志（不上传 Git）
```

---

## 技术栈

| 类别 | 技术选型 |
|------|----------|
| 基础 | Python、Jieba 分词、Pandas |
| 机器学习 | Scikit-learn、TF-IDF、RandomForest |
| 深度学习 | PyTorch、Transformers、MacBERT、LoRA (PEFT) |
| 模型压缩 | ONNX、动态量化 (INT8) |
| 工程服务 | FastAPI、Uvicorn |

---

## 环境安装

```bash
# 克隆项目
git clone <repo-url>
cd ContractGuard-NLP

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

> **GPU 训练（可选）**：如需训练 MacBERT + LoRA，请安装 CUDA 版 PyTorch：
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 快速开始

```bash
# 查看帮助
python main.py --help

# 1. 训练基线模型（TF-IDF + 随机森林）
python main.py --task train_baseline

# 2. 训练深度模型（MacBERT + LoRA，需 GPU）
python main.py --task train_lora

# 3. 压缩模型（量化 + ONNX 导出）
python main.py --task compress

# 4. 启动 Web 服务（浏览器访问 http://127.0.0.1:8000）
python main.py --task serve

# 5. 命令行单条推理
python main.py --task predict --text "乙方交付验收完成后180日内完成全款支付"
```

---

## API 接口

启动 `python main.py --task serve` 后：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 前端页面 |
| `/predict` | POST | 风险分类推理 |
| `/health` | GET | 服务健康检查 |

**推理请求示例：**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "乙方应在交货后30日内完成付款"}'
```

**响应示例：**

```json
{
  "text": "乙方应在交货后30日内完成付款",
  "risk_type": "payment_risk",
  "risk_label_id": 1,
  "risk_level": "高风险",
  "confidence": 0.9234,
  "confidence_array": [0.05, 0.92, 0.01, 0.01, 0.005, 0.005]
}
```

---

## 配置说明

所有超参数集中在 `config/config.yaml`：

| 配置段 | 说明 |
|--------|------|
| `data` | 数据路径、划分比例 |
| `risk_labels` | 6 类风险标签定义 |
| `model` | 模型选择、预训练路径 |
| `train.baseline` | 基线模型参数 |
| `train.lora` | LoRA 超参（r、lr、epochs、gamma） |
| `inference` | 推理设备、`model_type` 切换 |
| `api` | FastAPI 服务配置 |

---

## 性能要求

- **推理性能**：单条条款 ≤ 200ms，百条合同批量 ≤ 10s
- **模型精度**：核心风险类别（付款/合规风险）召回率 ≥ 92%

---

## License

MIT License
