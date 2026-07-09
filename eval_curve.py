# -*- coding: utf-8 -*-
"""同一 reference、多帧平均评测曲线（降低单图评测方差，用于调参时稳定比较）。

思路：取某场景（默认 5x5x4/0/npy）前 N 帧，逐帧 raw→log1p→模型→expm1，对同一 reference 算
PSNR/MSSIM/r，画曲线 + 给 mean±std。可选传入 --baseline_checkpoint（如 N2N），在**同样 50 帧**
上再跑一遍，报**配对 ΔPSNR 的 mean±std**（配对抵消场景噪声，是方差最小的比较）。

注意：默认场景来自 level4（训练分布内），此曲线衡量的是 in-distribution 去噪质量，方差低、
适合调参排名；**不等于泛化**。泛化仍须用留出的 level1-OOD、多场景评测。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer_eval_robust import load2d, infer, metrics          # 复用同一套加载/推理/指标口径
from models.denoiser_feats import DenoiserWithFeats
from utils.checkpoint import load_weights_flexible


def natural_key(p: Path):
    """按文件名里的数字自然排序（frame2 在 frame10 前面）。"""
    m = re.findall(r"\d+", p.stem)
    return (int(m[0]) if m else 0, p.stem)


def load_model(ckpt, device):
    m = DenoiserWithFeats(input_channels=1).to(device).eval()
    print(f"[INFO] load {ckpt}:", load_weights_flexible(m, ckpt, device))
    return m


def run_curve(model, frames, ref, dr, device):
    ps, ss, rs = [], [], []
    for fp in frames:
        out = infer(model, load2d(fp), device)
        p, s, r = metrics(out, ref, dr)
        ps.append(p); ss.append(s); rs.append(r)
    return np.array(ps), np.array(ss), np.array(rs)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dr = float(args.max)

    ref = load2d(args.reference)
    scene = Path(args.scene_dir)
    frames = sorted([p for p in scene.glob("*.npy")], key=natural_key)[:args.n_frames]
    if not frames:
        raise FileNotFoundError(f"{scene} 下没找到 .npy 帧")
    print(f"[INFO] reference={args.reference} shape={ref.shape}; 场景={scene} 取前 {len(frames)} 帧 "
          f"(data_range={dr:g})")

    p1, s1, r1 = run_curve(load_model(args.checkpoint, device), frames, ref, dr, device)
    have_base = bool(args.baseline_checkpoint)
    if have_base:
        p0, s0, r0 = run_curve(load_model(args.baseline_checkpoint, device), frames, ref, dr, device)

    # ---- 逐帧 CSV ----
    hdr = "frame,robust_psnr,robust_mssim,robust_r"
    cols = [ [fp.stem for fp in frames], p1, s1, r1 ]
    if have_base:
        hdr += ",base_psnr,base_mssim,base_r,dPSNR"
        cols += [p0, s0, r0, p1 - p0]
    with open(out_dir / "per_frame.csv", "w") as f:
        f.write(hdr + "\n")
        for i in range(len(frames)):
            f.write(",".join(str(c[i]) if isinstance(c[i], str) else f"{c[i]:.4f}" for c in cols) + "\n")

    # ---- 汇总 ----
    def stat(name, a):
        print(f"  {name:>12}: mean={a.mean():.3f}  std={a.std(ddof=1):.3f}  "
              f"min={a.min():.3f}  max={a.max():.3f}")
    print(f"\n=== Robust-N2N（{len(frames)} 帧）===")
    stat("PSNR", p1); stat("MSSIM", s1); stat("r", r1)
    if have_base:
        print(f"\n=== N2N baseline（同 {len(frames)} 帧）===")
        stat("PSNR", p0); stat("MSSIM", s0); stat("r", r0)
        d = p1 - p0                                            # 配对差：抵消场景噪声
        print(f"\n=== 配对 ΔPSNR = Robust - N2N（逐帧）===")
        print(f"  mean={d.mean():+.3f}  std={d.std(ddof=1):.3f}  "
              f"赢={int((d > 0).sum())}/{len(d)} 帧  "
              f"→ {'Robust 稳定更好' if d.mean() > d.std(ddof=1) else '差异在噪声内、分不开'}")

    # ---- 曲线 ----
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    x = np.arange(len(frames))
    ax.plot(x, p1, "-o", ms=3, label=f"Robust-N2N (mean {p1.mean():.3f})", color="#D4537E")
    ax.axhline(p1.mean(), ls="--", lw=1, color="#D4537E", alpha=0.6)
    if have_base:
        ax.plot(x, p0, "-o", ms=3, label=f"N2N baseline (mean {p0.mean():.3f})", color="#378ADD")
        ax.axhline(p0.mean(), ls="--", lw=1, color="#378ADD", alpha=0.6)
    ax.set_xlabel("frame index"); ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"per-frame PSNR vs same reference  ({scene.parent.parent.name}/{scene.parent.name}, {len(frames)} frames)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "psnr_curve.png", bbox_inches="tight"); plt.close(fig)
    print(f"\n[OK] -> {out_dir}/psnr_curve.png, per_frame.csv")


def parse_args():
    p = argparse.ArgumentParser(description="多帧平均评测曲线（同一 reference，降方差调参尺子）")
    p.add_argument("--checkpoint", required=True, help="待评测模型 checkpoint")
    p.add_argument("--baseline_checkpoint", default="", help="可选：基线(如 N2N)，同帧配对对比")
    p.add_argument("--scene_dir", default="/mnt2/songyd/5x5/5x5x4/0/npy", help="帧所在目录（内含 .npy）")
    p.add_argument("--n_frames", type=int, default=50)
    p.add_argument("--reference", default="/home/songyd/Projects/Robust-N2N/reference.npy")
    p.add_argument("--max", type=float, default=255.0, help="PSNR/MSSIM 的 data_range")
    p.add_argument("--out_dir", default="results/eval_curve")
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
