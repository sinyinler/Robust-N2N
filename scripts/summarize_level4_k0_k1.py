#!/usr/bin/env python3
"""汇总 W1/K0/K1 在 Level4 scene0 前 500 帧上的未校准配对指标。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIGS = {
    "W1": "Gaussian默认，EMA=0.996，feature warmup=20%，weight=0.10，patch=16",
    "K0": "Gaussian默认，EMA=0.996，feature warmup=20%，weight=0.05，patch=8",
    "K1": "Gaussian=0，EMA=0.999，feature warmup=20%，weight=0.05，patch=8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_root",
        default="results/eval_paper/level4_K0_K1_E100_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"评估CSV不存在：{path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"评估CSV为空：{path}")
    required = {
        "frame",
        "robust_psnr",
        "robust_mssim",
        "robust_r",
        "base_psnr",
        "base_mssim",
        "base_r",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"{path} 缺少字段：{sorted(missing)}")
    return rows


def values(rows: list[dict[str, str]], prefix: str) -> dict[str, np.ndarray]:
    return {
        metric: np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows])
        for metric in ("psnr", "mssim", "r")
    }


def main() -> None:
    args = parse_args()
    root = Path(args.eval_root)
    rows = {
        tag: load_rows(root / f"level4_seen_{tag}_vs_W1_s{args.seed}" / "per_frame.csv")
        for tag in ("K0", "K1")
    }

    frames_k0 = [row["frame"] for row in rows["K0"]]
    frames_k1 = [row["frame"] for row in rows["K1"]]
    if frames_k0 != frames_k1:
        raise RuntimeError("K0/K1 的逐帧顺序不一致，不能进行配对比较")

    w1_k0 = values(rows["K0"], "base")
    w1_k1 = values(rows["K1"], "base")
    for metric in ("psnr", "mssim", "r"):
        if not np.allclose(w1_k0[metric], w1_k1[metric], atol=5e-5, rtol=0.0):
            raise RuntimeError(f"两次评估中的 W1 {metric} 不一致")

    curves = {
        "W1": w1_k0,
        "K0": values(rows["K0"], "robust"),
        "K1": values(rows["K1"], "robust"),
    }
    n_frames = len(frames_k0)
    records: list[dict[str, object]] = []
    for tag in ("W1", "K0", "K1"):
        psnr = curves[tag]["psnr"]
        delta_w1 = psnr - curves["W1"]["psnr"]
        delta_k0 = psnr - curves["K0"]["psnr"]
        records.append(
            {
                "method": tag,
                "configuration": CONFIGS[tag],
                "psnr": float(psnr.mean()),
                "ssim": float(curves[tag]["mssim"].mean()),
                "r": float(curves[tag]["r"].mean()),
                "delta_psnr_vs_W1": float(delta_w1.mean()),
                "wins_vs_W1": int((delta_w1 > 0).sum()),
                "delta_psnr_vs_K0": float(delta_k0.mean()),
                "wins_vs_K0": int((delta_k0 > 0).sum()),
                "n_frames": n_frames,
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"K0_K1_vs_W1_s{args.seed}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "| 方法 | Level4 PSNR | SSIM | r | ΔPSNR vs W1 | win/W1 | ΔPSNR vs K0 | win/K0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        tag = str(record["method"])
        delta_k0_text = "-" if tag == "K0" else f"{record['delta_psnr_vs_K0']:+.3f}"
        wins_k0_text = "-" if tag == "K0" else f"{record['wins_vs_K0']}/{n_frames}"
        lines.append(
            f"| {tag} | {record['psnr']:.3f} | {record['ssim']:.4f} | "
            f"{record['r']:.4f} | {record['delta_psnr_vs_W1']:+.3f} | "
            f"{record['wins_vs_W1']}/{n_frames} | {delta_k0_text} | {wins_k0_text} |"
        )
    lines.extend(
        [
            "",
            f"> 指标为 seed{args.seed}、epoch100、Level4 scene0 前500帧相对同一reference的均值；",
            "> 均为未经仿射校准的原始模型输出，500帧不能当作500个独立场景。",
        ]
    )
    markdown_path = root / f"K0_K1_vs_W1_s{args.seed}_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 将三条逐帧曲线画在同一坐标系，便于定位组合配置的退化区间。
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    x = np.arange(n_frames)
    colors = {"W1": "#378ADD", "K0": "#E07A2D", "K1": "#4A9D67"}
    for tag in ("W1", "K0", "K1"):
        psnr = curves[tag]["psnr"]
        ax.plot(x, psnr, lw=1.0, color=colors[tag], alpha=0.85,
                label=f"{tag} (mean {psnr.mean():.3f})")
    ax.set_xlabel("frame index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Level4 scene0 first 500 frames: W1 vs K0 vs K1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    curve_path = root / f"K0_K1_vs_W1_s{args.seed}_psnr_curve.png"
    fig.savefig(curve_path, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(lines))
    print(f"\n[OK] CSV: {csv_path}")
    print(f"[OK] Markdown: {markdown_path}")
    print(f"[OK] PSNR curve: {curve_path}")


if __name__ == "__main__":
    main()
