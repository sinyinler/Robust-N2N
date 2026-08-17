#!/usr/bin/env python3
"""Summarize the H4/J0/J1/J2 Level4 paired evaluation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METHODS = {
    "H4": "log域Gaussian（当前Level4最优）",
    "J0": "raw域Gamma only",
    "J1": "25%区域内patch级Gamma/Gaussian 50/50互斥混合",
    "J2": "同区域Gamma→Gaussian，二者强度×1/sqrt(2)",
}


def load(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["frame"]: row for row in csv.DictReader(handle)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_root",
        default="results/eval_paper/level4_hybrid_H4_E100_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.eval_root)
    candidate_rows = {
        tag: load(root / f"level4_seen_{tag}_vs_H4_s{args.seed}" / "per_frame.csv")
        for tag in ("J0", "J1", "J2")
    }
    frames = list(candidate_rows["J0"])
    if len(frames) != 500:
        raise RuntimeError(f"expected 500 frames, got {len(frames)}")

    reference_base = candidate_rows["J0"]
    h4_psnr = np.array([float(reference_base[f]["base_psnr"]) for f in frames])
    h4_ssim = np.array([float(reference_base[f]["base_mssim"]) for f in frames])
    h4_r = np.array([float(reference_base[f]["base_r"]) for f in frames])
    results = {
        "H4": {
            "psnr": float(h4_psnr.mean()),
            "ssim": float(h4_ssim.mean()),
            "r": float(h4_r.mean()),
            "delta": 0.0,
            "wins": None,
        }
    }

    for tag, rows in candidate_rows.items():
        if list(rows) != frames:
            raise RuntimeError(f"frame order mismatch for {tag}")
        base = np.array([float(rows[f]["base_psnr"]) for f in frames])
        if not np.allclose(base, h4_psnr, atol=5e-5):
            raise RuntimeError(f"H4 baseline mismatch in {tag}")
        psnr = np.array([float(rows[f]["robust_psnr"]) for f in frames])
        results[tag] = {
            "psnr": float(psnr.mean()),
            "ssim": float(np.mean([float(rows[f]["robust_mssim"]) for f in frames])),
            "r": float(np.mean([float(rows[f]["robust_r"]) for f in frames])),
            "delta": float((psnr - base).mean()),
            "wins": int((psnr > base).sum()),
        }

    records = []
    for tag in ("H4", "J0", "J1", "J2"):
        row = results[tag]
        records.append({
            "method": tag,
            "corruption": METHODS[tag],
            "psnr": row["psnr"],
            "ssim": row["ssim"],
            "r": row["r"],
            "delta_psnr_vs_H4": row["delta"],
            "wins_vs_H4": row["wins"],
            "n_frames": len(frames),
        })

    csv_path = root / "level4_hybrid_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    maxima = {
        metric: max(results[tag][metric] for tag in results)
        for metric in ("psnr", "ssim", "r")
    }

    def fmt(tag: str, metric: str) -> str:
        digits = 3 if metric == "psnr" else 4
        value = results[tag][metric]
        text = f"{value:.{digits}f}"
        return f"**{text}**" if np.isclose(value, maxima[metric]) else text

    lines = [
        "# Level4 Gamma/Gaussian联合扰动消融",
        "",
        "| 方法 | 辅助分支扰动 | PSNR | SSIM | r | ΔPSNR vs H4 | win/H4 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for tag in ("H4", "J0", "J1", "J2"):
        row = results[tag]
        wins = "-" if row["wins"] is None else f"{row['wins']}/{len(frames)}"
        lines.append(
            f"| {tag} | {METHODS[tag]} | {fmt(tag, 'psnr')} | "
            f"{fmt(tag, 'ssim')} | {fmt(tag, 'r')} | "
            f"{row['delta']:+.3f} | {wins} |"
        )
    lines.extend([
        "",
        "> seed42、epoch100、Level4 scene0前500帧；指标来自未经仿射校准的原始输出。",
    ])
    md_path = root / "level4_hybrid_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] CSV -> {csv_path}")
    print(f"[OK] Markdown -> {md_path}")


if __name__ == "__main__":
    main()
