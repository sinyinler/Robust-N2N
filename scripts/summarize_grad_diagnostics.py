# -*- coding: utf-8 -*-
"""汇总 train_masked.py 生成的 grad_diagnostics.jsonl。"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    "n2n_norm",
    "weighted_feature_norm",
    "feature_to_n2n_ratio",
    "cosine",
)


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if float(record.get("ramp", 0.0)) >= args.min_ramp]
    if not records:
        raise RuntimeError(
            f"{input_path} 中没有 ramp >= {args.min_ramp:g} 的记录；"
            "可降低 --min_ramp 或增加诊断频率。"
        )

    scale_names = sorted({name for record in records for name in record["scales"]})
    summaries = []
    for scale in scale_names:
        items = [record["scales"][scale] for record in records if scale in record["scales"]]
        row = {
            "scale": scale,
            "n_records": len(items),
            "negative_cosine_fraction": float(
                sum(float(item["cosine"]) < 0.0 for item in items) / len(items)
            ),
            "strong_conflict_fraction": float(
                sum(float(item["cosine"]) < args.conflict_threshold for item in items) / len(items)
            ),
        }
        details = {}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            details[metric] = describe(values)
            for statistic_name, value in details[metric].items():
                row[f"{metric}_{statistic_name}"] = value
        summaries.append((row, details))

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "grad_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0][0].keys()))
        writer.writeheader()
        writer.writerows(row for row, _ in summaries)

    payload = {
        "input": str(input_path),
        "min_ramp": args.min_ramp,
        "conflict_threshold": args.conflict_threshold,
        "n_source_records": len(records),
        "scales": {
            row["scale"]: {
                "n_records": row["n_records"],
                "negative_cosine_fraction": row["negative_cosine_fraction"],
                "strong_conflict_fraction": row["strong_conflict_fraction"],
                **details,
            }
            for row, details in summaries
        },
    }
    json_path = out_dir / "grad_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'scale':>10} | {'feat/n2n':>10} | {'cosine':>9} | {'cos<0':>8} | {'cos<thr':>8}")
    print("-" * 58)
    for row, _ in summaries:
        print(
            f"{row['scale']:>10} | {row['feature_to_n2n_ratio_mean']:>10.4f} | "
            f"{row['cosine_mean']:>+9.4f} | {row['negative_cosine_fraction']:>8.1%} | "
            f"{row['strong_conflict_fraction']:>8.1%}"
        )
    print(f"\n[OK] {csv_path}\n[OK] {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 masked feature 分目标梯度诊断")
    parser.add_argument("--input", required=True, help="grad_diagnostics.jsonl")
    parser.add_argument("--out_dir", default="", help="默认写入 input 同目录")
    parser.add_argument("--min_ramp", type=float, default=0.99, help="忽略 feature warmup 早期记录")
    parser.add_argument("--conflict_threshold", type=float, default=-0.2)
    args = parser.parse_args()
    if not 0.0 <= args.min_ramp <= 1.0:
        raise ValueError("min_ramp 必须位于 [0,1]")
    return args


if __name__ == "__main__":
    main(parse_args())
