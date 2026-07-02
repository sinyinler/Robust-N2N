from __future__ import annotations

"""测试 BSN 在 T 翻译(白化)后的图上是否可行。

用零样本 ZS-N2N（Mansour & Heckel, CVPR'23）作 BSN 家族代表：靠"对角 2×2 下采样配对"
做单图自监督，**前提是相邻像素噪声独立(白)**。所以：
  - 原始散斑(空间相关) → BSN 在 raw 上去不动（配对视图共享相关噪声）；
  - T 把噪声白化后(lag1 0.82→0.13) → BSN 应该能有效去噪。

对每张输入并排：noisy / BSN(raw) / T(I) 翻译图 / BSN(translated) / [NTN=D'(T)] / [reference]，
灰度+jet 两排；给了 reference 则在 log1p 域算 PSNR/MSSIM/r。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.denoiser import Denoiser
from models.ntn import NoiseTranslator
from utils.checkpoint import load_weights_flexible
from utils.io import read_image_any
from utils.metrics import center_crop_to_match

try:
    from skimage.metrics import structural_similarity as _sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def log1p_np(x):
    return np.log1p(np.clip(x, 0, None)).astype(np.float32)


def metrics(img, ref, dr):
    a, b = center_crop_to_match(img, ref)
    if HAVE_SK:
        p = float(_sk_psnr(b, a, data_range=dr)); s = float(_sk_ssim(b, a, data_range=dr))
    else:
        p = s = float("nan")
    x = a.ravel() - a.mean(); y = b.ravel() - b.mean()
    r = float((x * y).sum() / ((np.sqrt((x * x).sum() * (y * y).sum())) or 1.0))
    return p, s, r


# ---------------- ZS-N2N（单图自监督 BSN） ----------------
def pair_downsampler(img):
    c = img.shape[1]
    f1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=img.device).repeat(c, 1, 1, 1)
    f2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=img.device).repeat(c, 1, 1, 1)
    return F.conv2d(img, f1, stride=2, groups=c), F.conv2d(img, f2, stride=2, groups=c)


class ZSNet(nn.Module):
    def __init__(self, c=1, ch=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, ch, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, c, 1))

    def forward(self, x):
        return self.net(x)


def zsn2n(noisy2d, device, iters=1500, lr=1e-3):
    """对单张 2D 图(已在 log 域)做 ZS-N2N 去噪，返回同域结果。"""
    x = torch.from_numpy(noisy2d).float()[None, None].to(device)
    lo, hi = float(x.min()), float(x.max()); rng = (hi - lo) or 1.0
    xn = (x - lo) / rng  # 归一到 ~[0,1] 训练更稳
    net = ZSNet(1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(iters):
        d1, d2 = pair_downsampler(xn)
        p1 = d1 - net(d1); p2 = d2 - net(d2)
        loss_res = 0.5 * (F.mse_loss(d1, p2) + F.mse_loss(d2, p1))
        den = xn - net(xn)
        dd1, dd2 = pair_downsampler(den)
        loss_cons = 0.5 * (F.mse_loss(p1, dd1) + F.mse_loss(p2, dd2))
        opt.zero_grad(); (loss_res + loss_cons).backward(); opt.step()
    with torch.no_grad():
        out = (xn - net(xn)) * rng + lo
    return out[0, 0].cpu().numpy().astype(np.float32)


def pad_mult(x, m=32):
    h, w = x.shape[-2:]; ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, ph, pw


@torch.no_grad()
def apply_model(model, z, device):
    t = torch.from_numpy(z).float()[None, None].to(device)
    tp, ph, pw = pad_mult(t)
    out = model(tp)
    if ph: out = out[..., :-ph, :]
    if pw: out = out[..., :-pw]
    return out[0, 0].cpu().numpy().astype(np.float32)


def out_stem(path):
    parts = list(path.parts)
    tail = parts[-4:-1] + [path.stem] if path.parent.name in ("npy", "lbf") else parts[-3:-1] + [path.stem]
    return "_".join(p.replace(":", "").replace("\\", "_").replace("/", "_") for p in tail)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    translator = NoiseTranslator(input_channels=1, width=args.width, middle_blocks=args.middle_blocks,
                                 inject_sigma=args.inject_sigma, residual_scale=args.residual_scale).to(device).eval()
    print("[INFO] T:", load_weights_flexible(translator, args.translator_checkpoint, device))
    expert = None
    if args.gaussian_expert_checkpoint:
        expert = Denoiser(input_channels=1).to(device).eval()
        print("[INFO] D':", load_weights_flexible(expert, args.gaussian_expert_checkpoint, device))

    ref_z = None
    if args.reference:
        ref_z = log1p_np(read_image_any(args.reference))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for item in args.inputs:
        path = Path(item)
        z = log1p_np(read_image_any(path))
        translated = apply_model(translator, z, device)
        print(f"[INFO] {path}: ZS-N2N on raw ...")
        bsn_raw = zsn2n(z, device, iters=args.iters)
        print(f"[INFO] {path}: ZS-N2N on translated ...")
        bsn_tr = zsn2n(translated, device, iters=args.iters)
        panels = [("noisy", z), ("BSN(raw)", bsn_raw), ("T(I)", translated), ("BSN(translated)", bsn_tr)]
        if expert is not None:
            panels.append(("NTN=D'(T)", apply_model(expert, translated, device)))
        if ref_z is not None:
            panels.append(("reference", ref_z))
            dr = float(ref_z.max() - ref_z.min()) or 1.0
            print(f"     metrics(log1p) vs reference (P/MSSIM/r):")
            for name, img in panels[:-1]:
                p, s, r = metrics(img, ref_z, dr)
                print(f"       {name:16s}  {p:7.2f} / {s:.3f} / {r:.3f}")

        # 出图：2 行(灰度/jet) × N 列，共用参考(或第一张)窗位
        base = ref_z if ref_z is not None else z
        vmin, vmax = np.percentile(base, [1, 99])
        n = len(panels)
        fig, ax = plt.subplots(2, n, figsize=(3.3 * n, 7), dpi=110)
        for j, (name, img) in enumerate(panels):
            h = min(img.shape[0], base.shape[0]); w = min(img.shape[1], base.shape[1])
            ax[0, j].imshow(img[:h, :w], cmap="gray", vmin=vmin, vmax=vmax); ax[0, j].set_title(name, fontsize=10)
            ax[1, j].imshow(img[:h, :w], cmap=args.cmap, vmin=vmin, vmax=vmax)
            for r_ in (0, 1):
                ax[r_, j].set_xticks([]); ax[r_, j].set_yticks([])
        ax[0, 0].set_ylabel("grayscale"); ax[1, 0].set_ylabel(args.cmap)
        fig.suptitle(f"{out_stem(path)}  (log1p domain)", fontsize=11)
        fig.tight_layout()
        op = out_dir / f"{out_stem(path)}_bsn.png"
        fig.savefig(op, bbox_inches="tight"); plt.close(fig)
        print(f"[INFO] -> {op}")


def parse_args():
    p = argparse.ArgumentParser(description="Test single-image BSN (ZS-N2N) on T-translated (whitened) BFI.")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--translator_checkpoint", required=True)
    p.add_argument("--gaussian_expert_checkpoint", default="", help="可选：加一列 NTN=D'(T) 对照")
    p.add_argument("--reference", default="", help="可选：给了则在 log1p 域算 PSNR/MSSIM/r")
    p.add_argument("--iters", type=int, default=1500, help="ZS-N2N 每张单图训练迭代数")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--middle_blocks", type=int, default=2)
    p.add_argument("--inject_sigma", type=float, default=1.0)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--cmap", default="jet")
    p.add_argument("--out_dir", default="results/images/bsn_test")
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
