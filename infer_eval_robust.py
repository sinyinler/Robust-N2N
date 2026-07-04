# -*- coding: utf-8 -*-
"""对单张 raw.npy 跑 Robust-N2N 推理，并与 reference.npy 及 N2N 结果(ours.npy) 对比。

流程：raw → log1p → RobustDenoiser(推理，GIBlock 关闭) → expm1 → 去噪图(raw 尺度)。
指标：PSNR / MSSIM(data_range=255) + Pearson r，分别对 Robust-N2N 输出 与 ours.npy(N2N) 计算。
可视化：每种结果出一张四联图 [去噪前灰度 | 去噪前 jet | 去噪后灰度 | 去噪后 jet]。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.robust_denoiser import RobustDenoiser
from utils.checkpoint import load_weights_flexible


def load2d(p):
    a = np.squeeze(np.load(p)).astype(np.float32)
    if a.ndim != 2:
        raise ValueError(f"{p} 不是 2D：shape={a.shape}")
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def center_crop(a, h, w):
    t = max(0, (a.shape[0] - h) // 2); l = max(0, (a.shape[1] - w) // 2)
    return a[t:t + h, l:l + w]


def pearson(a, b):
    x = a.ravel() - a.mean(); y = b.ravel() - b.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum())) or 1.0
    return float((x * y).sum() / d)


def metrics(img, ref, dr):
    h = min(img.shape[0], ref.shape[0]); w = min(img.shape[1], ref.shape[1])
    a, b = center_crop(img, h, w), center_crop(ref, h, w)
    return (float(sk_psnr(b, a, data_range=dr)),
            float(sk_ssim(b, a, data_range=dr)),
            pearson(a, b))


@torch.no_grad()
def infer(model, raw, device):
    """raw(2D, raw 尺度) → log1p → model → expm1 → 去噪图(raw 尺度)。"""
    z = np.log1p(np.clip(raw, 0, None)).astype(np.float32)
    t = torch.from_numpy(z)[None, None].to(device)
    h, w = t.shape[-2:]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    out = model(t)
    if ph:
        out = out[..., :-ph, :]
    if pw:
        out = out[..., :-pw]
    out = out.squeeze().cpu().numpy().astype(np.float32)
    return np.expm1(out)  # 回到 raw 尺度


def quad_vis(noisy, denoised, title, path, cmap="jet"):
    """四联图：去噪前灰度 | 去噪前 jet | 去噪后灰度 | 去噪后 jet（每格自百分位窗位）。"""
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6), dpi=140)
    panels = [("noisy (gray)", noisy, "gray"), ("noisy (jet)", noisy, cmap),
              ("denoised (gray)", denoised, "gray"), ("denoised (jet)", denoised, cmap)]
    for j, (name, img, cm) in enumerate(panels):
        lo, hi = np.percentile(img, [1, 99])
        if hi <= lo:
            lo, hi = float(img.min()), float(img.max() or 1.0)
        ax[j].imshow(img, cmap=cm, vmin=lo, vmax=hi); ax[j].set_title(name, fontsize=11)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dr = float(args.max)

    raw = load2d(args.raw)
    ref = load2d(args.reference)

    model = RobustDenoiser(input_channels=1).to(device).eval()  # eval → GIBlock 注入关闭
    print("[INFO] load:", load_weights_flexible(model, args.checkpoint, device))
    robust = infer(model, raw, device)
    np.save(out_dir / "robust_out.npy", robust)

    rows = [("Robust-N2N", robust)]
    if args.n2n:
        rows.append(("N2N (ours)", load2d(args.n2n)))

    print(f"\nreference = {args.reference}  (data_range={dr:g})")
    print(f"{'method':>12} | {'PSNR':>8} | {'MSSIM':>7} | {'r':>7}")
    print("-" * 44)
    for name, img in rows:
        p, s, r = metrics(img, ref, dr)
        print(f"{name:>12} | {p:>8.3f} | {s:>7.4f} | {r:>7.4f}")

    # 四联图：Robust-N2N（以及 N2N，若给了 ours.npy）
    quad_vis(raw, robust, "raw  vs  Robust-N2N", out_dir / "quad_robust.png", args.cmap)
    if args.n2n:
        quad_vis(raw, load2d(args.n2n), "raw  vs  N2N (ours)", out_dir / "quad_n2n.png", args.cmap)
    print(f"\n[INFO] -> {out_dir}/robust_out.npy, quad_robust.png" + (", quad_n2n.png" if args.n2n else ""))


def parse_args():
    p = argparse.ArgumentParser(description="Robust-N2N inference + metrics vs reference.")
    p.add_argument("--checkpoint", required=True, help="Robust-N2N checkpoint (.pth)")
    p.add_argument("--raw", default="/home/songyd/Projects/Robust-N2N/raw.npy")
    p.add_argument("--reference", default="/home/songyd/Projects/Robust-N2N/reference.npy")
    p.add_argument("--n2n", default="/home/songyd/Projects/Robust-N2N/ours.npy", help="N2N 已有结果(ours.npy)；空则不比")
    p.add_argument("--max", type=float, default=255.0, help="PSNR/MSSIM 的 data_range")
    p.add_argument("--cmap", default="jet")
    p.add_argument("--out_dir", default="results/infer_robust")
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
