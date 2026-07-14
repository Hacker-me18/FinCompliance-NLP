# -*- coding: utf-8 -*-
"""
将 processed/ 下三个 jsonl 文件合并为单个 CSV，方便 pandas 读取后划分数据集。

schema（与 dataset/schema.py 一致，加上 source / confidence）：
  id      自增编号
  text    条款原文
    label   标签（字符 '0'~'5' / 'review'）
  source  数据来源（bid / court_judge / synthetic）
  confidence 标签置信度

输出 CSV 放在 data/processed/contract_dataset.csv。
"""
import json
import csv
from pathlib import Path


def merge(output_path: str = None) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    files = [
        ("bid_contract_samples.jsonl", "bid"),
        ("cj_processed.jsonl", "court_judge"),
        ("syn_processed.jsonl", "synthetic"),
    ]

    # CSV 列顺序与项目 schema 对齐
    schema = ["id", "text", "label", "source", "confidence"]
    if output_path is None:
        output_path = str(processed_dir / "contract_dataset.csv")

    rows = []
    for fname, src_name in files:
        p = processed_dir / fname
        if not p.exists():
            print(f"跳过缺失文件: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rows.append({
                    "text": item["text"],
                    "label": item["label"],
                    "source": src_name,
                    "confidence": item.get("confidence", 0.75),
                })

    # 打乱顺序：按 source 分批入队，再交替合并，保证三类来源在 CSV 中均匀交错
    from collections import defaultdict
    import random
    random.seed(42)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["source"]].append(r)
    for k in buckets:
        random.shuffle(buckets[k])

    # 按 bid > court_judge > synthetic 顺序交替取（轮询），
    # 某个桶耗尽就跳过它，直到全部耗尽
    ordered = []
    keys = [s for _, s in files]                    # ['bid','court_judge','synthetic']
    iters = {k: iter(buckets[k]) for k in keys}
    exhausted = set()
    while len(exhausted) < len(keys):
        for k in keys:
            if k in exhausted:
                continue
            try:
                ordered.append(next(iters[k]))
            except StopIteration:
                exhausted.add(k)

    # 写入 CSV，加 id 列
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema)
        writer.writeheader()
        for idx, row in enumerate(ordered, 1):
            writer.writerow({"id": idx, **row})

    # 统计
    from collections import Counter
    total = len(ordered)
    by_label = Counter(r["label"] for r in ordered)
    by_source = Counter(r["source"] for r in ordered)
    print(f"写出到: {output_path}")
    print(f"总样本 {total}")
    print("来源分布:", dict(by_source))
    print("标签分布:", dict(sorted(by_label.items())))
    return {"total": total, "labels": dict(by_label), "sources": dict(by_source)}


if __name__ == "__main__":
    merge()
