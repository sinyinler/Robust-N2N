from __future__ import annotations

"""对若干张图相对同一 reference 计算 PSNR / MSSIM / r，并出灰度+伪彩对比图。

与模型无关，只吃 .npy（也支持 .lbf 若装了项目依赖）。指标：
- PSNR：峰值信噪比，data_range 由 --max 指定（这里 255）。
- MSSIM：mean SSIM（skimage 窗口化 SSIM 的均值），data_range=--max。
- r：Pearson 相关系数（展平后与 reference 的线性相关）。
所有图先中心裁剪到公共尺寸再比，保证逐像素对齐。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{path} 不是 2D 图：shape={arr.shape}")
    return np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def center_crop_to(a: np.ndarray, h: int, w: int) -> np.ndarray:
    t = max(0, (a.shape[0] - h) // 2)
    l = max(0, (a.shape[1] - w) // 2)
    return a[t:t + h, l:l + w]


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
    return float((a * b).sum() / denom)


def parse_images(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for it in items:
        if "=" in it:
            label, path = it.split("=", 1)
        else:
            label, path = Path(it).stem, it
        out.append((label.strip(), path.strip()))
    return out


def main(args) -> None:
    dr = float(args.max)
    ref = load_npy(args.reference)
    pairs = parse_images(args.images)
    imgs = [(lab, load_npy(p)) for lab, p in pairs]

    # 公共尺寸（含 reference）取最小，逐张中心裁剪对齐。
    H = min([ref.shape[0]] + [im.shape[0] for _, im in imgs])
    W = min([ref.shape[1]] + [im.shape[1] for _, im in imgs])
    ref_c = center_crop_to(ref, H, W)
    imgs_c = [(lab, center_crop_to(im, H, W)) for lab, im in imgs]

    # 可选：在 log1p 域算指标（模型工作域），抑制 raw 域亮血管(指数还原)对 PSNR 的放大，更公平。
    if args.log_domain:
        ref_c = np.log1p(np.clip(ref_c, 0, None))
        imgs_c = [(lab, np.log1p(np.clip(im, 0, None))) for lab, im in imgs_c]
        dr = float(ref_c.max() - ref_c.min()) or 1.0  # log1p 域用 reference 自身范围作 data_range

    # 可选 winsorize：从 reference 取 [clip_pct, 100-clip_pct] 作「同一个」裁剪区间，
    # 对 reference 与所有待比图统一 clip，抑制 hot/dead/饱和像素对 PSNR、r 的不成比例影响。
    clip_lo = clip_hi = None
    if args.clip_pct > 0:
        clip_lo, clip_hi = np.percentile(ref_c, [args.clip_pct, 100 - args.clip_pct])
        clip_lo, clip_hi = float(clip_lo), float(clip_hi)
        ref_c = np.clip(ref_c, clip_lo, clip_hi)
        imgs_c = [(lab, np.clip(im, clip_lo, clip_hi)) for lab, im in imgs_c]
        dr = (clip_hi - clip_lo) or 1.0  # data_range 与裁剪区间自洽，避免 PSNR 虚高

    rows = []
    clip_note = (f", winsorize[{args.clip_pct}%,{100-args.clip_pct}%]->[{clip_lo:.3g},{clip_hi:.3g}]"
                 if clip_lo is not None else "")
    print(f"\nreference = {args.reference}  (size {H}x{W}, data_range={dr:g}{clip_note})")
    print(f"{'image':>10} | {'PSNR(dB)':>9} | {'MSSIM':>7} | {'r':>7}")
    print("-" * 42)
    for lab, im in imgs_c:
        psnr = float(sk_psnr(ref_c, im, data_range=dr))
        mssim = float(sk_ssim(ref_c, im, data_range=dr))
        r = pearson_r(ref_c, im)
        rows.append({"image": lab, "psnr": psnr, "mssim": mssim, "r": r})
        print(f"{lab:>10} | {psnr:>9.3f} | {mssim:>7.4f} | {r:>7.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps({"reference": args.reference, "data_range": dr, "size": [H, W],
                    "clip_pct": args.clip_pct, "clip_range": [clip_lo, clip_hi],
                    "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 可视化：上排灰度、下排伪彩；列 = reference + 各图。共享窗宽窗位，便于公平比较 ----
    panels = imgs_c + [("reference", ref_c)]  # 顺序：noisy, n2n, ntn, ..., reference（参考放最后）
    vmin = float(np.percentile(ref_c, args.pclip)); vmax = float(np.percentile(ref_c, 100 - args.pclip))
    if vmax <= vmin:
        vmin, vmax = float(ref_c.min()), float(ref_c.max() or 1.0)
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.8), dpi=150)
    if n == 1:
        axes = axes.reshape(2, 1)
    metric_by_label = {r["image"]: r for r in rows}
    for j, (lab, im) in enumerate(panels):
        title = lab if lab == "reference" else (
            f"{lab}\nPSNR {metric_by_label[lab]['psnr']:.2f} | MSSIM {metric_by_label[lab]['mssim']:.3f} | r {metric_by_label[lab]['r']:.3f}")
        axes[0, j].imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(title, fontsize=10)
        axes[1, j].imshow(im, cmap=args.cmap, vmin=vmin, vmax=vmax)
        for r_ in (0, 1):
            axes[r_, j].set_xticks([]); axes[r_, j].set_yticks([])
    axes[0, 0].set_ylabel("grayscale", fontsize=11)
    axes[1, 0].set_ylabel(f"pseudo-color ({args.cmap})", fontsize=11)
    fig.suptitle(f"vs reference (data_range={dr:g})", fontsize=12)
    fig.tight_layout()
    combo = out_dir / "compare_gray_color.png"
    fig.savefig(combo, bbox_inches="tight"); plt.close(fig)

    # 另存每张的灰度/伪彩单图，便于放大
    for lab, im in panels:
        for tag, cmap in (("gray", "gray"), ("color", args.cmap)):
            p = out_dir / f"{lab}_{tag}.png"
            plt.imsave(p, np.clip(im, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax)

    print(f"\n[INFO] 指标 -> {out_dir/'metrics.json'}")
    print(f"[INFO] 对比图(灰度+伪彩) -> {combo}")
    print(f"[INFO] 单图 -> {out_dir}/<label>_gray.png / _color.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute PSNR/MSSIM/r vs a reference; render grayscale + pseudo-color.")
    p.add_argument("--reference", required=True, help="参考图 .npy")
    p.add_argument("--images", nargs="+", required=True,
                   help="待比较图，格式 label=path（如 t5=/path/t5-0.npy n2n=/path/n2n.npy ntn=/path/ntn.npy）")
    p.add_argument("--max", type=float, default=255.0, help="data_range（PSNR/SSIM 用），默认 255；开启 --clip_pct 时自动改用裁剪区间宽度")
    p.add_argument("--clip_pct", type=float, default=0.0,
                   help="winsorize 百分位（如 0.5 表示裁到 [0.5%%,99.5%%]，从 reference 取、对所有图统一）。0=不裁。")
    p.add_argument("--log_domain", type=int, default=0,
                   help="1=在 log1p 域算指标（消除 raw 亮血管对 PSNR 的指数放大，更公平）；data_range 自动用 log 域范围。")
    p.add_argument("--cmap", type=str, default="turbo", help="伪彩 colormap（turbo/jet/viridis...）")
    p.add_argument("--pclip", type=float, default=1.0, help="显示窗位百分位裁剪（取 reference 的 p..100-p）")
    p.add_argument("--out_dir", type=str, default="results/metrics_vs_ref")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
