from __future__ import annotations

"""对一批 flat 场景（如 mix/191..226，每个 <scene>/npy 下多帧）批量做 N2N vs NTN 泛化评估。

每个场景：参考 = 自身多帧(raw 域)均值；输入 = 第 --frame 帧；在 log1p 域(模型工作域)
对 noisy / n2n / ntn 算 PSNR/MSSIM/r（data_range 用该场景参考的 log 域范围）。最后汇总均值。
"""

import argparse
import json
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

from data.discovery import list_supported_files, load_2d
from models.denoiser import Denoiser
from models.ntn import NoiseTranslator
from utils.checkpoint import load_weights_flexible
from utils.metrics import center_crop_to_match

try:
    from skimage.metrics import structural_similarity as _sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as _sk_psnr

    def psnr(a, b, dr):
        a, b = center_crop_to_match(a, b); return float(_sk_psnr(a, b, data_range=dr))

    def ssim(a, b, dr):
        a, b = center_crop_to_match(a, b); return float(_sk_ssim(a, b, data_range=dr))
    SSIM = "skimage"
except Exception:
    from utils.metrics import psnr as _p, ssim_simple as _s
    psnr = lambda a, b, dr: float(_p(a, b, data_range=dr))
    ssim = lambda a, b, dr: float(_s(a, b, data_range=dr))
    SSIM = "simple"


def pearson(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
    return float((a * b).sum() / d)


def log1p_np(x):
    return np.log1p(np.clip(x, 0, None)).astype(np.float32)


def pad_mult(x, m=32):
    h, w = x.shape[-2:]; ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, ph, pw


def crop_back(x, ph, pw):
    if ph: x = x[..., :-ph, :]
    if pw: x = x[..., :-pw]
    return x


@torch.no_grad()
def denoise(z, device, n2n, translator, expert):
    t = torch.from_numpy(z).float().unsqueeze(0).unsqueeze(0).to(device)
    tp, ph, pw = pad_mult(t)
    n2n_z = crop_back(n2n(tp), ph, pw).squeeze().cpu().numpy()
    ntn_z = crop_back(expert(translator(tp)), ph, pw).squeeze().cpu().numpy()
    return n2n_z, ntn_z


def save_vis(scene, noisy_z, n2n_z, ntn_z, ref_z, m, out_dir, cmap):
    """每个场景出一张 2 行(灰度/jet) × 4 列(noisy/n2n/ntn/ref) 对比图，共用参考窗位。"""
    h = min(x.shape[0] for x in (noisy_z, n2n_z, ntn_z, ref_z))
    w = min(x.shape[1] for x in (noisy_z, n2n_z, ntn_z, ref_z))
    panels = [("noisy\nP%.1f S%.2f r%.2f" % tuple(m["noisy"]), noisy_z[:h, :w]),
              ("n2n\nP%.1f S%.2f r%.2f" % tuple(m["n2n"]), n2n_z[:h, :w]),
              ("ntn\nP%.1f S%.2f r%.2f" % tuple(m["ntn"]), ntn_z[:h, :w]),
              ("reference", ref_z[:h, :w])]
    fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=110)
    for j, (name, img) in enumerate(panels):
        # 每个面板用各自的百分位窗位：避免 NTN 因全局尺度偏移而显示成一团黑
        vmin, vmax = np.percentile(img, [1, 99])
        if vmax <= vmin:
            vmin, vmax = float(img.min()), float(img.max() or 1.0)
        ax[0, j].imshow(img, cmap="gray", vmin=vmin, vmax=vmax); ax[0, j].set_title(name, fontsize=10)
        ax[1, j].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        for r in (0, 1):
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
    ax[0, 0].set_ylabel("grayscale"); ax[1, 0].set_ylabel(cmap)
    fig.suptitle(f"scene {scene} (log1p domain)", fontsize=12)
    fig.tight_layout()
    p = Path(out_dir) / "vis"; p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / f"{scene}.png", bbox_inches="tight"); plt.close(fig)


def scene_list(args):
    if args.scenes:
        return [str(s) for s in args.scenes]
    if args.scene_range:
        a, b = args.scene_range
        return [str(s) for s in range(a, b + 1)]
    raise SystemExit("需要 --scenes 或 --scene_range")


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    n2n = Denoiser(input_channels=1).to(device).eval()
    print("[INFO] N2N:", load_weights_flexible(n2n, args.n2n_checkpoint, device))
    expert = Denoiser(input_channels=1).to(device).eval()
    print("[INFO] D':", load_weights_flexible(expert, args.gaussian_expert_checkpoint, device))
    translator = NoiseTranslator(input_channels=1, width=args.width, middle_blocks=args.middle_blocks,
                                 inject_sigma=args.inject_sigma, residual_scale=args.residual_scale).to(device).eval()
    print("[INFO] T:", load_weights_flexible(translator, args.translator_checkpoint, device))
    print(f"[INFO] SSIM backend={SSIM}, log1p domain metrics")

    rows = {"noisy": [], "n2n": [], "ntn": []}
    per_scene = []
    for sc in scene_list(args):
        folder = Path(args.root) / sc / args.data_subdir
        files = list_supported_files(folder)
        if len(files) < 2:
            print(f"[skip] {sc}: <2 frames"); continue
        ref_files = files[:args.ref_frames] if args.ref_frames > 0 else files
        ref_raw = np.mean([load_2d(f) for f in ref_files], axis=0).astype(np.float32)
        ref_z = log1p_np(ref_raw)
        dr = float(ref_z.max() - ref_z.min()) or 1.0

        idx = min(args.frame, len(files) - 1)
        noisy_z = log1p_np(load_2d(files[idx]))
        n2n_z, ntn_z = denoise(noisy_z, device, n2n, translator, expert)

        m = {}
        for k, img in (("noisy", noisy_z), ("n2n", n2n_z), ("ntn", ntn_z)):
            a_, b_ = center_crop_to_match(img, ref_z)
            if args.affine:
                # 最小二乘拟合 b_ ≈ k1*a_ + k0，去掉全局增益/偏移（N2N、NTN 同样处理）
                x = a_.ravel().astype(np.float64); y = b_.ravel().astype(np.float64)
                k1 = ((x - x.mean()) * (y - y.mean())).sum() / (((x - x.mean()) ** 2).sum() or 1.0)
                a_ = (k1 * a_ + (y.mean() - k1 * x.mean())).astype(np.float32)
            p, s, r = psnr(a_, b_, dr), ssim(a_, b_, dr), pearson(a_, b_)
            rows[k].append([p, s, r]); m[k] = [p, s, r]
        per_scene.append({"scene": sc, **m})
        print(f"  {sc}: n2n {m['n2n'][0]:.2f}/{m['n2n'][1]:.3f}/{m['n2n'][2]:.3f}  "
              f"ntn {m['ntn'][0]:.2f}/{m['ntn'][1]:.3f}/{m['ntn'][2]:.3f}")
        if args.vis and (args.max_vis <= 0 or len(per_scene) <= args.max_vis):
            save_vis(sc, noisy_z, n2n_z, ntn_z, ref_z, m, args.out_dir, args.cmap)

    if not per_scene:
        raise SystemExit("没有可评估的场景")

    summ = {k: np.mean(rows[k], axis=0).tolist() for k in rows}
    print(f"\n==== 汇总 ({len(per_scene)} scenes, log1p 域, 每场景自身多帧均值为参考) ====")
    print(f"{'method':>10} | {'PSNR':>7} | {'MSSIM':>6} | {'r':>6}")
    print("-" * 38)
    lab = {"noisy": "noisy", "n2n": "N2N", "ntn": "NTN(ours)"}
    for k in ("noisy", "n2n", "ntn"):
        print(f"{lab[k]:>10} | {summ[k][0]:>7.3f} | {summ[k][1]:>6.4f} | {summ[k][2]:>6.4f}")
    print("-" * 38)
    print(f"{'NTN-N2N':>10} | {summ['ntn'][0]-summ['n2n'][0]:>+7.3f} | "
          f"{summ['ntn'][1]-summ['n2n'][1]:>+6.4f} | {summ['ntn'][2]-summ['n2n'][2]:>+6.4f}")

    out = {"n_scenes": len(per_scene), "ssim": SSIM,
           "summary": {lab[k]: {"psnr": summ[k][0], "mssim": summ[k][1], "r": summ[k][2]} for k in rows},
           "gain_ntn_minus_n2n": {"psnr": summ['ntn'][0]-summ['n2n'][0], "mssim": summ['ntn'][1]-summ['n2n'][1],
                                  "r": summ['ntn'][2]-summ['n2n'][2]},
           "per_scene": per_scene}
    op = Path(args.out_dir); op.mkdir(parents=True, exist_ok=True)
    (op / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] -> {op/'metrics.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Batch N2N vs NTN eval over flat scenes (per-scene multi-frame mean ref).")
    p.add_argument("--root", default="/mnt2/songyd/mix")
    p.add_argument("--data_subdir", default="npy")
    p.add_argument("--scenes", type=str, nargs="*", default=None, help="显式场景编号列表")
    p.add_argument("--scene_range", type=int, nargs=2, default=None, help="区间，如 --scene_range 191 226")
    p.add_argument("--frame", type=int, default=0, help="去噪该帧索引（默认 0）")
    p.add_argument("--ref_frames", type=int, default=0, help="参考用多少帧均值；0=全部")
    p.add_argument("--n2n_checkpoint", required=True)
    p.add_argument("--translator_checkpoint", required=True)
    p.add_argument("--gaussian_expert_checkpoint", required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--middle_blocks", type=int, default=2)
    p.add_argument("--inject_sigma", type=float, default=1.0)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--out_dir", default="results/eval_ood_scenes")
    p.add_argument("--vis", type=int, default=1, help="是否每个场景出对比图(1/0)，边跑边存到 out_dir/vis/")
    p.add_argument("--max_vis", type=int, default=0, help="最多出多少张图；<=0 表示全部场景都出")
    p.add_argument("--cmap", default="jet")
    p.add_argument("--affine", type=int, default=0,
                   help="1=对每张图做最小二乘仿射对齐(去全局增益/偏移)再算指标，分离“结构”与“尺度”。")
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
