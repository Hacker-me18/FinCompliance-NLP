# -*- coding: utf-8 -*-
"""
ContractGuard-NLP 程序统一入口
制造业合同财务风险智能识别系统 - 命令行任务调度

任务类型（--task）：
  data           数据建设：采集 + 清洗 + 弱监督标注 + LLM 合成 + 合并划分
  train_baseline Stage1 基线模型训练（TF-IDF + 随机森林）
  train_fasttext Stage2 轻量模型训练（FastText）
  train_lora     Stage3 深度模型微调（MacBERT/RoBERTa + LoRA，线上主模型）
  evaluate       模型评估（Recall / F1 / Precision / Accuracy）
  predict        单条/批量风险分类推理
  explain        LLM 风险解释（成因/影响/整改建议/等级）
  report         生成标准化财务风险审核报告
  serve          启动 FastAPI 推理服务

使用示例：
  python main.py --task data
  python main.py --task train_lora
  python main.py --task predict --text "乙方交付验收完成后180日内完成全款支付"
  python main.py --task serve
"""

import argparse
import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载全局配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_data_pipeline(config: dict):
    """
    数据建设全流程（Day2）
    公开数据采集 → 司法案例采集 → 文本清洗 → 弱监督初标注 → LLM 批量合成 → 数据集合并划分
    """
    raise NotImplementedError("数据建设流程待实现")


def run_train_baseline(config: dict):
    """Stage1：TF-IDF + 随机森林 基线模型训练（仅用于实验对比，不参与线上推理）"""
    from src.train.stage1_baseline import train_baseline
    train_baseline(config)


def run_train_fasttext(config: dict):
    """Stage2：FastText 轻量模型训练（过渡优化模型）"""
    from src.train.stage2_fasttext import train_fasttext
    train_fasttext(config)


def run_train_lora(config: dict):
    """Stage3：MacBERT/RoBERTa + LoRA 微调（线上主模型）"""
    from src.train.stage3_macbert_lora import train_lora
    train_lora(config)


def run_evaluate(config: dict):
    """模型评估：Recall(核心) / F1 / Precision / Accuracy + 混淆矩阵"""
    raise NotImplementedError("评估流程待实现")


def run_predict(config: dict, text: str = None, file_path: str = None):
    """
    风险分类推理
    - 单条条款推理：--text "乙方交付验收完成后180日内完成全款支付"
    - 批量合同推理：--file "path/to/contract.pdf"
    """
    from src.inference.predict import predict_text, predict_file
    if text:
        result = predict_text(config, text)
        print(f"输入条款：{text}")
        print(f"识别结果：{result}")
    elif file_path:
        results = predict_file(config, file_path)
        print(f"合同文件：{file_path}")
        print(f"识别结果：{results}")
    else:
        print("请通过 --text 或 --file 指定推理输入")


def run_explain(config: dict, text: str = None):
    """LLM 风险解释：成因 / 影响 / 整改建议 / 风险等级"""
    from src.llm_enhance.risk_explainer import explain_risk
    if text:
        explanation = explain_risk(config, text)
        print(f"输入条款：{text}")
        print(f"风险解释：{explanation}")
    else:
        print("请通过 --text 指定需要解释的条款")


def run_report(config: dict, file_path: str = None):
    """生成标准化财务风险审核报告"""
    from src.report.report_generator import generate_report
    if file_path:
        report = generate_report(config, file_path)
        print(f"报告已生成：{report}")
    else:
        print("请通过 --file 指定合同文件路径")


def run_compress(config: dict):
    """模型压缩：训练好的模型 → 量化 + ONNX"""
    from src.compress.model_compressor import compress_model
    compress_model(
        model_dir=str(Path(config["train"]["lora"]["output_dir"])),
        output_dir=str(Path("models/compressed")),
        model_name=config["model"]["backbone"],
    )


def run_serve(config: dict):
    """启动 FastAPI 推理服务"""
    import uvicorn
    from demo.api import create_app
    app = create_app(config)
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])


def main():
    parser = argparse.ArgumentParser(
        description="ContractGuard-NLP 制造业合同财务风险智能识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
任务类型说明：
  data           数据建设全流程
  train_baseline Stage1 基线模型（TF-IDF + 随机森林）
  train_fasttext Stage2 轻量模型（FastText）
  train_lora     Stage3 深度模型（MacBERT + LoRA，线上主模型）
  evaluate       模型评估
  predict        风险分类推理（--text 或 --file）
  explain        LLM 风险解释（--text）
  report         报告生成（--file）
  serve          启动 FastAPI 服务
        """
    )

    parser.add_argument(
        "--task", type=str, required=True,
        choices=["data", "train_baseline", "train_fasttext", "train_lora",
                 "evaluate", "predict", "explain", "report", "compress", "serve"],
        help="执行任务类型"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="配置文件路径（默认 config/config.yaml）")
    parser.add_argument("--text", type=str, default=None,
                        help="单条条款文本（用于 predict / explain）")
    parser.add_argument("--file", type=str, default=None,
                        help="合同文件路径（用于 predict / report）")

    args = parser.parse_args()
    config = load_config(args.config)

    # 任务路由
    if args.task == "data":
        run_data_pipeline(config)
    elif args.task == "train_baseline":
        run_train_baseline(config)
    elif args.task == "train_fasttext":
        run_train_fasttext(config)
    elif args.task == "train_lora":
        run_train_lora(config)
    elif args.task == "evaluate":
        run_evaluate(config)
    elif args.task == "predict":
        run_predict(config, text=args.text, file_path=args.file)
    elif args.task == "explain":
        run_explain(config, text=args.text)
    elif args.task == "report":
        run_report(config, file_path=args.file)
    elif args.task == "compress":
        run_compress(config)
    elif args.task == "serve":
        run_serve(config)


if __name__ == "__main__":
    main()
