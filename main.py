"""FinCompliance-NLP 程序统一入口"""
import argparse
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="FinCompliance-NLP 金融合规对话意图识别")
    parser.add_argument("--task", type=str, required=True,
                        choices=["process", "train", "evaluate", "predict", "compress"],
                        help="执行任务: process/train/evaluate/predict/compress")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.task == "process":
        print("数据处理模块待实现")
    elif args.task == "train":
        print("模型训练模块待实现")
    elif args.task == "evaluate":
        print("模型评估模块待实现")
    elif args.task == "predict":
        print("推理预测模块待实现")
    elif args.task == "compress":
        print("模型压缩模块待实现")


if __name__ == "__main__":
    main()
