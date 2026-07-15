# -*- coding: utf-8 -*-
"""批量评估 Masked N2N 的 A/C 多 seed、多 epoch checkpoint。

本脚本固定使用同一组 level4 帧和同一 reference，逐帧计算 A-base 与
C-feature 的 PSNR/MSSIM/Pearson r，并输出：

- ``seed_epoch_summary.csv``：每个 seed/epoch 的均值、标准差与配对增益；
- ``epoch_summary.csv``：跨 seed 的 epoch 曲线；
- ``per_frame.csv``：所有逐帧配对结果；
- ``summary.json``：完整机器可读结果；
- ``epoch_curve.png``：A/C 指标与 C-A 增益曲线；
- ``compare/``：同一窗宽下的全图和中心局部放大。

ID 指标用于诊断学习曲线，不应直接作为 test-set 选 checkpoint 的依据；是否延长训练
应优先结合 ``history.jsonl`` 中的 validation loss 判断。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from infer_eval_robust import infer, load2d, metrics
from models.masked_denoiser import MaskedDenoiserWithFeats
from utils.checkpoint import load_weights_flexible


def natural_key(path: Path):
    """按文件名数字自然排序，确保 frame2 位于 frame10 之前。"""

    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[0]) if numbers else 0, path.stem)


def sample_std(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def summary_stats(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": sample_std(values),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def load_validation_loss(checkpoint: Path, epoch: int) -> float | None:
    """读取 checkpoint 同目录 history.jsonl 中对应 epoch 的 validation loss。"""

    history = checkpoint.parent / "history.jsonl"
    if not history.is_file():
        return None
    matched = None
    for line in history.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("epoch", -1)) == int(epoch):
            value = record.get("val")
            matched = None if value is None else float(value)
    return matched


def bootstrap_mean_ci(values, repeats: int, seed: int) -> tuple[float, float] | None:
    """对逐帧配对差的均值做 percentile bootstrap。"""

    values = np.asarray(values, dtype=np.float64)
    if repeats <= 0 or values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def evaluate_checkpoint(
    checkpoint: Path,
    frames: list[np.ndarray],
    reference: np.ndarray,
    data_range: float,
    device: torch.device,
    max_vis_frames: int,
    strict_load: bool,
) -> tuple[dict[str, np.ndarray], list[np.ndarray], dict[str, int]]:
    """一次加载一个模型，避免 18 个 checkpoint 同时占用 GPU 内存。"""

    model = MaskedDenoiserWithFeats(image_channels=1).to(device).eval()
    load_info = load_weights_flexible(model, str(checkpoint), device)
    if strict_load and (load_info["loaded"] <= 0 or load_info["skipped"] != 0):
        raise RuntimeError(f"checkpoint 未完整加载：{checkpoint} -> {load_info}")

    psnr_values, ssim_values, r_values = [], [], []
    visual_outputs: list[np.ndarray] = []
    for index, raw in enumerate(frames):
        output = infer(model, raw, device)
        p, s, r = metrics(output, reference, data_range)
        psnr_values.append(p)
        ssim_values.append(s)
        r_values.append(r)
        if index < max_vis_frames:
            visual_outputs.append(output)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        {
            "psnr": np.asarray(psnr_values, dtype=np.float64),
            "mssim": np.asarray(ssim_values, dtype=np.float64),
            "r": np.asarray(r_values, dtype=np.float64),
        },
        visual_outputs,
        load_info,
    )


def center_crop(array: np.ndarray, height: int, width: int) -> np.ndarray:
    top = max(0, (array.shape[0] - height) // 2)
    left = max(0, (array.shape[1] - width) // 2)
    return array[top:top + height, left:left + width]


def normalize_u8(array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = np.clip((array.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def save_comparison(
    raw: np.ndarray,
    arm_a: np.ndarray,
    arm_c: np.ndarray,
    reference: np.ndarray,
    path: Path,
    zoom_size: int,
) -> None:
    """保存同窗宽全图与中心局部放大，检查细小结构是否被磨平。"""

    named = [("noisy", raw), ("A-base", arm_a), ("C-feature", arm_c), ("reference", reference)]
    height = min(array.shape[0] for _, array in named)
    width = min(array.shape[1] for _, array in named)
    named = [(name, center_crop(array, height, width)) for name, array in named]
    values = np.concatenate([array.reshape(-1) for _, array in named])
    vmin, vmax = np.percentile(values, [1.0, 99.0])

    zoom = min(int(zoom_size), height, width)
    top = max(0, (height - zoom) // 2)
    left = max(0, (width - zoom) // 2)
    title_height = 24
    canvas = Image.new("RGB", (width * len(named), height * 2 + title_height), color=(0, 0, 0))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)

    for column, (name, array) in enumerate(named):
        panel = Image.fromarray(normalize_u8(array, float(vmin), float(vmax)), mode="L").convert("RGB")
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle(
            [left, top, left + zoom - 1, top + zoom - 1],
            outline=(255, 0, 0),
            width=max(1, width // 256),
        )
        zoom_panel = panel.crop((left, top, left + zoom, top + zoom)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
        x0 = column * width
        draw.text((x0 + 5, 5), name, fill=(255, 255, 0), font=font)
        canvas.paste(panel, (x0, title_height))
        canvas.paste(zoom_panel, (x0, title_height + height))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def checkpoint_path(args, arm: str, seed: int, epoch: int) -> Path:
    template = args.a_dir_template if arm == "A" else args.c_dir_template
    directory = Path(args.checkpoint_root) / template.format(seed=seed)
    return directory / f"model_epoch_{epoch}.pth"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_epoch_curves(epoch_rows: list[dict], path: Path) -> None:
    epochs = np.asarray([row["epoch"] for row in epoch_rows])
    a_psnr = np.asarray([row["a_psnr_mean"] for row in epoch_rows])
    c_psnr = np.asarray([row["c_psnr_mean"] for row in epoch_rows])
    a_std = np.asarray([row["a_psnr_seed_std"] for row in epoch_rows])
    c_std = np.asarray([row["c_psnr_seed_std"] for row in epoch_rows])
    delta = np.asarray([row["delta_psnr_mean"] for row in epoch_rows])
    delta_std = np.asarray([row["delta_psnr_seed_std"] for row in epoch_rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    axes[0].errorbar(epochs, a_psnr, yerr=a_std, marker="o", capsize=3, label="A-base")
    axes[0].errorbar(epochs, c_psnr, yerr=c_std, marker="o", capsize=3, label="C-feature")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("ID PSNR (dB)")
    axes[0].set_title("mean across seeds (error bar = seed std)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].errorbar(epochs, delta, yerr=delta_std, marker="o", capsize=3, color="#D4537E")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("C - A PSNR (dB)")
    axes[1].set_title("paired gain across seeds")
    axes[1].grid(alpha=0.3)
    for x, y in zip(epochs, delta):
        axes[1].annotate(f"{y:+.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    scene_dir = Path(args.scene_dir)
    frame_paths = sorted(scene_dir.glob("*.npy"), key=natural_key)[:args.n_frames]
    if not frame_paths:
        raise FileNotFoundError(f"{scene_dir} 下未找到 .npy 帧")

    missing = [
        checkpoint_path(args, arm, seed, epoch)
        for seed in args.seeds for epoch in args.epochs for arm in ("A", "C")
        if not checkpoint_path(args, arm, seed, epoch).is_file()
    ]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"缺少 {len(missing)} 个 checkpoint：\n{preview}")

    reference = load2d(args.reference)
    frames = [load2d(path) for path in frame_paths]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] device={device} frames={len(frames)} seeds={args.seeds} epochs={args.epochs}\n"
        f"[INFO] scene={scene_dir} reference={args.reference} data_range={args.max:g}"
    )

    seed_epoch_records: list[dict] = []
    per_frame_rows: list[dict] = []
    for seed in args.seeds:
        for epoch in args.epochs:
            ckpt_a = checkpoint_path(args, "A", seed, epoch)
            ckpt_c = checkpoint_path(args, "C", seed, epoch)
            print(f"\n========== seed={seed} epoch={epoch} ==========")
            a_metrics, a_visuals, load_a = evaluate_checkpoint(
                ckpt_a, frames, reference, args.max, device,
                args.max_vis_frames, bool(args.strict_load),
            )
            c_metrics, c_visuals, load_c = evaluate_checkpoint(
                ckpt_c, frames, reference, args.max, device,
                args.max_vis_frames, bool(args.strict_load),
            )

            delta_psnr = c_metrics["psnr"] - a_metrics["psnr"]
            delta_mssim = c_metrics["mssim"] - a_metrics["mssim"]
            delta_r = c_metrics["r"] - a_metrics["r"]
            ci = bootstrap_mean_ci(
                delta_psnr,
                args.bootstrap_repeats,
                args.bootstrap_seed + seed * 100 + epoch,
            )

            record = {
                "seed": int(seed),
                "epoch": int(epoch),
                "checkpoint_a": str(ckpt_a),
                "checkpoint_c": str(ckpt_c),
                "load_a": load_a,
                "load_c": load_c,
                "val_a": load_validation_loss(ckpt_a, epoch),
                "val_c": load_validation_loss(ckpt_c, epoch),
                "a": {key: summary_stats(values) for key, values in a_metrics.items()},
                "c": {key: summary_stats(values) for key, values in c_metrics.items()},
                "delta": {
                    "psnr": summary_stats(delta_psnr),
                    "mssim": summary_stats(delta_mssim),
                    "r": summary_stats(delta_r),
                    "psnr_bootstrap_95ci": list(ci) if ci is not None else None,
                    "psnr_wins": int((delta_psnr > 0).sum()),
                    "n_frames": int(delta_psnr.size),
                },
            }
            seed_epoch_records.append(record)

            print(
                f"A={record['a']['psnr']['mean']:.3f}/{record['a']['mssim']['mean']:.4f} "
                f"C={record['c']['psnr']['mean']:.3f}/{record['c']['mssim']['mean']:.4f} "
                f"delta={record['delta']['psnr']['mean']:+.3f} dB "
                f"wins={record['delta']['psnr_wins']}/{len(frames)}"
            )

            for index, frame_path in enumerate(frame_paths):
                per_frame_rows.append({
                    "seed": seed,
                    "epoch": epoch,
                    "frame": frame_path.stem,
                    "frame_path": str(frame_path),
                    "a_psnr": float(a_metrics["psnr"][index]),
                    "c_psnr": float(c_metrics["psnr"][index]),
                    "delta_psnr": float(delta_psnr[index]),
                    "a_mssim": float(a_metrics["mssim"][index]),
                    "c_mssim": float(c_metrics["mssim"][index]),
                    "delta_mssim": float(delta_mssim[index]),
                    "a_r": float(a_metrics["r"][index]),
                    "c_r": float(c_metrics["r"][index]),
                    "delta_r": float(delta_r[index]),
                })

            for index, (a_output, c_output) in enumerate(zip(a_visuals, c_visuals)):
                save_comparison(
                    frames[index], a_output, c_output, reference,
                    out_dir / "compare" / f"seed{seed}_epoch{epoch}_{frame_paths[index].stem}.png",
                    args.zoom_size,
                )

    seed_epoch_csv = []
    for record in seed_epoch_records:
        ci = record["delta"]["psnr_bootstrap_95ci"]
        seed_epoch_csv.append({
            "seed": record["seed"],
            "epoch": record["epoch"],
            "val_a": record["val_a"],
            "val_c": record["val_c"],
            "a_psnr_mean": record["a"]["psnr"]["mean"],
            "c_psnr_mean": record["c"]["psnr"]["mean"],
            "delta_psnr_mean": record["delta"]["psnr"]["mean"],
            "delta_psnr_std": record["delta"]["psnr"]["std"],
            "delta_psnr_ci_low": None if ci is None else ci[0],
            "delta_psnr_ci_high": None if ci is None else ci[1],
            "psnr_wins": record["delta"]["psnr_wins"],
            "n_frames": record["delta"]["n_frames"],
            "a_mssim_mean": record["a"]["mssim"]["mean"],
            "c_mssim_mean": record["c"]["mssim"]["mean"],
            "delta_mssim_mean": record["delta"]["mssim"]["mean"],
            "a_r_mean": record["a"]["r"]["mean"],
            "c_r_mean": record["c"]["r"]["mean"],
            "delta_r_mean": record["delta"]["r"]["mean"],
        })

    epoch_records: list[dict] = []
    for epoch in args.epochs:
        selected = [record for record in seed_epoch_records if record["epoch"] == epoch]
        a_psnr = [record["a"]["psnr"]["mean"] for record in selected]
        c_psnr = [record["c"]["psnr"]["mean"] for record in selected]
        delta_psnr = [record["delta"]["psnr"]["mean"] for record in selected]
        valid_a = [record["val_a"] for record in selected if record["val_a"] is not None]
        valid_c = [record["val_c"] for record in selected if record["val_c"] is not None]
        epoch_records.append({
            "epoch": epoch,
            "a_psnr_mean": float(np.mean(a_psnr)),
            "a_psnr_seed_std": sample_std(a_psnr),
            "c_psnr_mean": float(np.mean(c_psnr)),
            "c_psnr_seed_std": sample_std(c_psnr),
            "delta_psnr_mean": float(np.mean(delta_psnr)),
            "delta_psnr_seed_std": sample_std(delta_psnr),
            "delta_psnr_min_seed": float(np.min(delta_psnr)),
            "delta_psnr_max_seed": float(np.max(delta_psnr)),
            "a_val_mean": None if not valid_a else float(np.mean(valid_a)),
            "c_val_mean": None if not valid_c else float(np.mean(valid_c)),
            "total_wins": int(sum(record["delta"]["psnr_wins"] for record in selected)),
            "total_frame_comparisons": int(sum(record["delta"]["n_frames"] for record in selected)),
        })

    write_csv(
        out_dir / "per_frame.csv", per_frame_rows,
        list(per_frame_rows[0].keys()),
    )
    write_csv(
        out_dir / "seed_epoch_summary.csv", seed_epoch_csv,
        list(seed_epoch_csv[0].keys()),
    )
    write_csv(
        out_dir / "epoch_summary.csv", epoch_records,
        list(epoch_records[0].keys()),
    )
    plot_epoch_curves(epoch_records, out_dir / "epoch_curve.png")

    payload = {
        "protocol": {
            "note": "ID learning-curve diagnostic; choose checkpoints by validation, not test PSNR.",
            "scene_dir": str(scene_dir),
            "reference": str(args.reference),
            "n_frames": len(frames),
            "data_range": float(args.max),
            "seeds": list(args.seeds),
            "epochs": list(args.epochs),
            "a_dir_template": args.a_dir_template,
            "c_dir_template": args.c_dir_template,
        },
        "epoch_summary": epoch_records,
        "seed_epoch_summary": seed_epoch_records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n========== 跨 seed epoch 汇总 ==========")
    print(f"{'epoch':>5} | {'A PSNR':>8} | {'C PSNR':>8} | {'C-A':>8} | {'wins':>9} | {'A val':>9} | {'C val':>9}")
    print("-" * 76)
    for row in epoch_records:
        a_val = "n/a" if row["a_val_mean"] is None else f"{row['a_val_mean']:.5f}"
        c_val = "n/a" if row["c_val_mean"] is None else f"{row['c_val_mean']:.5f}"
        print(
            f"{row['epoch']:>5} | {row['a_psnr_mean']:>8.3f} | {row['c_psnr_mean']:>8.3f} | "
            f"{row['delta_psnr_mean']:>+8.3f} | "
            f"{row['total_wins']:>3}/{row['total_frame_comparisons']:<5} | {a_val:>9} | {c_val:>9}"
        )
    print(f"\n[OK] 汇总与曲线写入 {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/C 多 seed、多 epoch 的 ID 配对评估")
    parser.add_argument("--checkpoint_root", default="results/checkpoints")
    parser.add_argument("--a_dir_template", default="maskfix_A_base_s{seed}")
    parser.add_argument("--c_dir_template", default="maskfix_C_feature_s{seed}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 187, 2413])
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--scene_dir", default="/mnt2/songyd/5x5/5x5x4/0/npy")
    parser.add_argument("--reference", default="/home/songyd/Projects/Robust-N2N/reference.npy")
    parser.add_argument("--n_frames", type=int, default=50)
    parser.add_argument("--max", type=float, default=255.0, help="PSNR/MSSIM data_range")
    parser.add_argument("--out_dir", default="results/eval_id/maskfix_epoch_sweep")
    parser.add_argument("--max_vis_frames", type=int, default=1)
    parser.add_argument("--zoom_size", type=int, default=128)
    parser.add_argument("--bootstrap_repeats", type=int, default=20_000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--strict_load", type=int, default=1)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    if not args.seeds or not args.epochs:
        raise ValueError("seeds 和 epochs 不能为空")
    if args.n_frames <= 0 or args.max_vis_frames < 0 or args.bootstrap_repeats < 0:
        raise ValueError("n_frames 必须为正；max_vis_frames/bootstrap_repeats 不能为负")
    if not math.isfinite(args.max) or args.max <= 0:
        raise ValueError("--max 必须是正数")
    return args


if __name__ == "__main__":
    main(parse_args())
