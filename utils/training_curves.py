# -*- coding: utf-8 -*-
"""从逐 epoch JSONL 历史记录生成训练曲线和便于检查的 CSV。"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_history(history_path: Path) -> list[dict[str, float]]:
    """读取完整 JSON 行；若训练中断留下半行，则跳过该行。"""

    records: list[dict[str, float]] = []
    if not history_path.is_file():
        return records
    for line_number, line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            print(f"[WARN] 跳过 history.jsonl 中无法解析的第 {line_number} 行")
            continue
        record: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                record[key] = float(value)
        if "epoch" in record:
            records.append(record)
    return records


def _write_csv(records: list[dict[str, float]], output_path: Path) -> None:
    """将 JSONL 中的所有数值字段同步成普通 CSV。"""

    preferred = [
        "epoch", "train", "total", "train_reconstruction", "val", "n2n",
        "weighted_mask_feature", "weighted_mask_pixel", "weighted_rtv",
        "mask_feature", "mask_pixel", "rtv", "hidden",
    ]
    all_keys = {key for record in records for key in record}
    fieldnames = [key for key in preferred if key in all_keys]
    fieldnames.extend(sorted(all_keys.difference(fieldnames)))
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _values(records: list[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    points = [(record["epoch"], record[key]) for record in records if key in record]
    return [point[0] for point in points], [point[1] for point in points]


def update_training_curves(history_path: str | Path, title: str) -> None:
    """更新 ``loss_curve.png`` 和 ``loss_history.csv``。

    原始 N2N 的 ``train``/``val`` 是同构目标。Masked N2N 的 ``total`` 含
    masked feature 等辅助项，而 ``val`` 只含重建与 RTV；因此额外生成
    ``train_reconstruction = n2n + weighted_rtv``，用于和 validation 公平对照。
    """

    history_path = Path(history_path)
    records = _read_history(history_path)
    if not records:
        return

    for record in records:
        if "n2n" in record and "weighted_rtv" in record:
            record["train_reconstruction"] = record["n2n"] + record["weighted_rtv"]

    output_dir = history_path.parent
    _write_csv(records, output_dir / "loss_history.csv")

    masked_history = any("total" in record for record in records)
    if masked_history:
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        main_ax, component_ax = axes
        main_series = [
            ("total", "train total", "#D55E00"),
            ("train_reconstruction", "train reconstruction (N2N+RTV)", "#0072B2"),
            ("val", "validation reconstruction (N2N+RTV)", "#009E73"),
        ]
        component_series = [
            ("n2n", "train N2N", "#0072B2"),
            ("weighted_mask_feature", "weighted feature", "#CC79A7"),
            ("weighted_mask_pixel", "weighted pixel", "#E69F00"),
            ("weighted_rtv", "weighted RTV", "#56B4E9"),
        ]
        for key, label, color in main_series:
            epochs, values = _values(records, key)
            if epochs:
                main_ax.plot(
                    epochs, values, marker="o", markersize=2.5, linewidth=1.6,
                    label=label, color=color,
                )
        for key, label, color in component_series:
            epochs, values = _values(records, key)
            if epochs and any(abs(value) > 0.0 for value in values):
                component_ax.plot(epochs, values, linewidth=1.5, label=label, color=color)
        main_ax.set_ylabel("Epoch-average loss")
        component_ax.set_ylabel("Training loss component")
        component_ax.set_xlabel("Epoch")
        main_ax.set_title(title)
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for key, label, color in (
            ("train", "train loss", "#0072B2"),
            ("val", "validation loss", "#D55E00"),
        ):
            epochs, values = _values(records, key)
            if epochs:
                ax.plot(
                    epochs, values, marker="o", markersize=3, linewidth=1.7,
                    label=label, color=color,
                )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Epoch-average loss")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.tight_layout()
    output_path = output_dir / "loss_curve.png"
    temp_path = output_dir / ".loss_curve.tmp.png"
    fig.savefig(temp_path, dpi=180, bbox_inches="tight", format="png")
    plt.close(fig)
    temp_path.replace(output_path)
