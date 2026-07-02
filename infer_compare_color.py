from __future__ import annotations

"""对每张输入跑 N2N 与 NTN(T->D')，输出「输入 / N2N / NTN」三联，灰度 + jet 两种显示。

每张输入生成一张 2 行 × 3 列的图：上排灰度、下排伪彩(jet)，列依次 Input / N2N / NTN。
三列共用同一窗宽窗位（取输入的百分位），对比公平。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
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
from utils.intensity import IntensityTransform
from utils.io import read_image_any


def pad_mult(x: torch.Tensor, m: int):
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, ph, pw


def crop_back(x, ph, pw):
    if ph:
        x = x[..., :-ph, :]
    if pw:
        x = x[..., :-pw]
    return x


@torch.no_grad()
def run(raw, transform, device, n2n, translator, expert, m):
    max_value = float(np.max(raw))
    z = transform.forward(torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0).to(device))
    zp, ph, pw = pad_mult(z, m)
    n2n_z = crop_back(n2n(zp), ph, pw)
    ntn_z = crop_back(expert(translator(zp)), ph, pw)
    out_n2n = transform.inverse(n2n_z, max_value=max_value).squeeze().cpu().numpy()
    out_ntn = transform.inverse(ntn_z, max_value=max_value).squeeze().cpu().numpy()
    return out_n2n.astype(np.float32), out_ntn.astype(np.float32)


def out_stem(path: Path) -> str:
    parts = list(path.parts)
    tail = parts[-4:-1] + [path.stem] if path.parent.name in ("npy", "lbf") else parts[-3:-1] + [path.stem]
    return "_".join(p.replace(":", "").replace("\\", "_").replace("/", "_") for p in tail)


def render(stem, raw, n2n, ntn, out_dir: Path, cmap: str, pclip: float):
    h = min(raw.shape[0], n2n.shape[0], ntn.shape[0])
    w = min(raw.shape[1], n2n.shape[1], ntn.shape[1])
    panels = [("Input", raw[:h, :w]), ("N2N", n2n[:h, :w]), ("NTN (ours)", ntn[:h, :w])]
    vmin = float(np.percentile(raw[:h, :w], pclip)); vmax = float(np.percentile(raw[:h, :w], 100 - pclip))
    if vmax <= vmin:
        vmin, vmax = float(raw.min()), float(raw.max() or 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9), dpi=130)
    for j, (name, img) in enumerate(panels):
        axes[0, j].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(name, fontsize=14)
        axes[1, j].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("grayscale", fontsize=13)
    axes[1, 0].set_ylabel(cmap, fontsize=13)
    fig.suptitle(stem, fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}_in_n2n_ntn.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = IntensityTransform(name=args.intensity_transform)
    n2n = Denoiser(input_channels=1).to(device).eval()
    print("[INFO] N2N:", load_weights_flexible(n2n, args.n2n_checkpoint, device))
    expert = Denoiser(input_channels=1).to(device).eval()
    print("[INFO] D':", load_weights_flexible(expert, args.gaussian_expert_checkpoint, device))
    translator = NoiseTranslator(input_channels=1, width=args.width, middle_blocks=args.middle_blocks,
                                 inject_sigma=args.inject_sigma, residual_scale=args.residual_scale).to(device).eval()
    print("[INFO] T:", load_weights_flexible(translator, args.translator_checkpoint, device))

    out_dir = Path(args.out_dir)
    for item in args.inputs:
        path = Path(item)
        raw = read_image_any(path).astype(np.float32)
        out_n2n, out_ntn = run(raw, transform, device, n2n, translator, expert, args.size_multiple)
        p = render(out_stem(path), raw, out_n2n, out_ntn, out_dir, args.cmap, args.pclip)
        print(f"[INFO] {path}  ->  {p}")


def parse_args():
    p = argparse.ArgumentParser(description="N2N vs NTN on inputs; show grayscale + colormap (Input/N2N/NTN).")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--n2n_checkpoint", required=True)
    p.add_argument("--translator_checkpoint", required=True)
    p.add_argument("--gaussian_expert_checkpoint", required=True)
    p.add_argument("--intensity_transform", choices=["none", "log1p", "boxcox", "learned_vst"], default="log1p")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--middle_blocks", type=int, default=2)
    p.add_argument("--inject_sigma", type=float, default=1.0)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--size_multiple", type=int, default=32)
    p.add_argument("--cmap", type=str, default="jet")
    p.add_argument("--pclip", type=float, default=1.0, help="显示窗位百分位裁剪（取输入的 p..100-p）。")
    p.add_argument("--out_dir", type=str, default="results/images/compare_color")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
