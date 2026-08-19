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
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer_eval_robust import load2d, infer, metrics          # 复用同一套加载/推理/指标口径
from models.denoiser_feats import DenoiserWithFeats
from models.masked_denoiser import MaskedDenoiserWithFeats
from utils.checkpoint import load_weights_flexible
from utils.photometric import fit_reference_affine


def natural_key(p: Path):
    """按文件名里的数字自然排序（frame2 在 frame10 前面）。"""
    m = re.findall(r"\d+", p.stem)
    return (int(m[0]) if m else 0, p.stem)


def load_model(ckpt, device, bias_free=False, masked_model=False):
    m = (MaskedDenoiserWithFeats(image_channels=1) if masked_model
         else DenoiserWithFeats(input_channels=1)).to(device)
    if bias_free:                                              # bias-free checkpoint 架构不同，须先改造再加载
        from models.bias_free import make_bias_free
        make_bias_free(m)
        m = m.to(device)                                       # BFBatchNorm2d 新建在 CPU，搬回 device
    m = m.eval()
    print(f"[INFO] load {ckpt} (bias_free={bool(bias_free)}, masked={bool(masked_model)}):",
          load_weights_flexible(m, ckpt, device))
    return m


def run_curve(model, frames, ref, dr, device, photometric_diagnostic=False):
    ps, ss, rs = [], [], []
    diagnostic = {
        "affine_a": [],
        "affine_b": [],
        "mean_ratio": [],
        "std_ratio": [],
        "affine_psnr": [],
        "affine_mssim": [],
        "affine_r": [],
        "affine_psnr_gain": [],
    } if photometric_diagnostic else None
    for fp in frames:
        out = infer(model, load2d(fp), device)
        p, s, r = metrics(out, ref, dr)
        ps.append(p); ss.append(s); rs.append(r)
        if diagnostic is not None:
            fit = fit_reference_affine(out, ref)
            corrected_p, corrected_s, corrected_r = metrics(fit.corrected, ref, dr)
            diagnostic["affine_a"].append(fit.scale)
            diagnostic["affine_b"].append(fit.offset)
            diagnostic["mean_ratio"].append(fit.mean_ratio)
            diagnostic["std_ratio"].append(fit.std_ratio)
            diagnostic["affine_psnr"].append(corrected_p)
            diagnostic["affine_mssim"].append(corrected_s)
            diagnostic["affine_r"].append(corrected_r)
            diagnostic["affine_psnr_gain"].append(corrected_p - p)
    curves = (np.array(ps), np.array(ss), np.array(rs))
    if diagnostic is None:
        # Keep the historical three-array return value for sweep_feat.py and
        # any external callers that use this helper without diagnostics.
        return curves
    diagnostic = {key: np.asarray(value) for key, value in diagnostic.items()}
    return (*curves, diagnostic)


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

    candidate_result = run_curve(
        load_model(args.checkpoint, device, args.bias_free, args.masked_model),
        frames, ref, dr, device, bool(args.photometric_diagnostic))
    if args.photometric_diagnostic:
        p1, s1, r1, diagnostic1 = candidate_result
    else:
        p1, s1, r1 = candidate_result
        diagnostic1 = None
    have_base = bool(args.baseline_checkpoint)
    if have_base:
        baseline_result = run_curve(
            load_model(args.baseline_checkpoint, device, args.baseline_bias_free,
                       args.baseline_masked_model),
            frames, ref, dr, device, bool(args.photometric_diagnostic))
        if args.photometric_diagnostic:
            p0, s0, r0, diagnostic0 = baseline_result
        else:
            p0, s0, r0 = baseline_result
            diagnostic0 = None
    else:
        diagnostic0 = None

    # ---- 逐帧 CSV ----
    hdr = "frame,robust_psnr,robust_mssim,robust_r"
    cols = [ [fp.stem for fp in frames], p1, s1, r1 ]
    if have_base:
        hdr += ",base_psnr,base_mssim,base_r,dPSNR"
        cols += [p0, s0, r0, p1 - p0]
    if diagnostic1 is not None:
        for name, values in diagnostic1.items():
            hdr += f",robust_{name}"
            cols.append(values)
        if have_base:
            for name, values in diagnostic0.items():
                hdr += f",base_{name}"
                cols.append(values)
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
        print("\n=== 配对 ΔPSNR = candidate - baseline（逐帧）===")
        print(f"  mean={d.mean():+.3f}  std={d.std(ddof=1):.3f}  "
              f"赢={int((d > 0).sum())}/{len(d)} 帧  "
              f"→ {'Robust 稳定更好' if d.mean() > d.std(ddof=1) else '差异在噪声内、分不开'}")

    def diagnostic_summary(name, values):
        summary = {key: float(array.mean()) for key, array in values.items()}
        print(f"\n=== {name} reference 仿射光度诊断（仅诊断，不是推理指标）===")
        print(f"  拟合 reference ≈ a·output+b: a={summary['affine_a']:.5f}, "
              f"b={summary['affine_b']:+.5f}")
        print(f"  output/reference: mean ratio={summary['mean_ratio']:.5f}, "
              f"std ratio={summary['std_ratio']:.5f}")
        print(f"  校正后 PSNR={summary['affine_psnr']:.3f} dB "
              f"(平均增益 {summary['affine_psnr_gain']:+.3f} dB), "
              f"MSSIM={summary['affine_mssim']:.4f}, r={summary['affine_r']:.4f}")
        return summary

    diagnostic_report = None
    if diagnostic1 is not None:
        diagnostic_report = {
            "warning": (
                "Reference-fitted diagnostic only. The affine-corrected metrics "
                "must not be reported as deployable inference or benchmark results."
            ),
            "formula": "reference ~= affine_a * output + affine_b",
            "candidate": diagnostic_summary("candidate", diagnostic1),
        }
        if diagnostic0 is not None:
            diagnostic_report["baseline"] = diagnostic_summary("baseline", diagnostic0)
        with open(out_dir / "photometric_diagnostic_summary.json", "w", encoding="utf-8") as f:
            json.dump(diagnostic_report, f, ensure_ascii=False, indent=2)

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
    if diagnostic1 is not None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=140, sharex=True)
        axes[0].plot(x, p1, "-", lw=1.2, label="candidate raw", color="#D4537E")
        axes[0].plot(x, diagnostic1["affine_psnr"], "--", lw=1.2,
                     label="candidate affine-corrected", color="#D4537E")
        if diagnostic0 is not None:
            axes[0].plot(x, p0, "-", lw=1.2, label="baseline raw", color="#378ADD")
            axes[0].plot(x, diagnostic0["affine_psnr"], "--", lw=1.2,
                         label="baseline affine-corrected", color="#378ADD")
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].set_title("Reference-fitted affine diagnostic (not an inference metric)")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].axhline(1.0, ls=":", lw=1, color="#666666")
        axes[1].plot(x, diagnostic1["mean_ratio"], "-", lw=1.2,
                     label="candidate mean ratio", color="#D4537E")
        axes[1].plot(x, diagnostic1["std_ratio"], "--", lw=1.2,
                     label="candidate std ratio", color="#A23B62")
        if diagnostic0 is not None:
            axes[1].plot(x, diagnostic0["mean_ratio"], "-", lw=1.2,
                         label="baseline mean ratio", color="#378ADD")
            axes[1].plot(x, diagnostic0["std_ratio"], "--", lw=1.2,
                         label="baseline std ratio", color="#245D91")
        axes[1].set_xlabel("frame index")
        axes[1].set_ylabel("output / reference ratio")
        axes[1].legend(ncol=2); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "photometric_diagnostic_curve.png", bbox_inches="tight")
        plt.close(fig)
    outputs = "psnr_curve.png, per_frame.csv"
    if diagnostic1 is not None:
        outputs += ", photometric_diagnostic_curve.png, photometric_diagnostic_summary.json"
    print(f"\n[OK] -> {out_dir}/[{outputs}]")


def parse_args():
    p = argparse.ArgumentParser(description="多帧平均评测曲线（同一 reference，降方差调参尺子）")
    p.add_argument("--checkpoint", required=True, help="待评测模型 checkpoint")
    p.add_argument("--baseline_checkpoint", default="", help="可选：基线(如 N2N)，同帧配对对比")
    p.add_argument("--scene_dir", default="/mnt2/songyd/5x5/5x5x4/0/npy", help="帧所在目录（内含 .npy）")
    p.add_argument("--n_frames", type=int, default=50)
    p.add_argument("--reference", default="/home/songyd/Projects/Robust-N2N/reference.npy")
    p.add_argument("--max", type=float, default=255.0, help="PSNR/MSSIM 的 data_range")
    p.add_argument("--out_dir", default="results/eval_curve")
    p.add_argument("--bias_free", type=int, default=0, help="主模型是否 Bias-Free 架构")
    p.add_argument("--baseline_bias_free", type=int, default=0, help="基线是否 Bias-Free 架构")
    p.add_argument("--masked_model", type=int, default=0,
                   help="1=主 checkpoint 来自 train_masked.py（模型内部推理时自动补全可见 mask）")
    p.add_argument("--baseline_masked_model", type=int, default=0,
                   help="1=基线 checkpoint 也来自 train_masked.py 的双通道公平基线")
    p.add_argument("--photometric_diagnostic", type=int, default=0,
                   help="1=用 reference 逐帧拟合 y_corrected=a*y+b，并报告校正前后指标（仅诊断）")
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
