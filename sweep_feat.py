# -*- coding: utf-8 -*-
"""跨视图特征一致性权重 sweep：一次铺完多条臂，跑完自动汇总成表。

臂的格式： w_feat:w_enc1,w_enc2,w_enc3,w_bottleneck
  - 权重为 0 的尺度**不参与**（不建 projector/predictor，省算力与显存）
  - 特征项实际总权重 = w_feat × 该尺度权重（代码里 L_feat = Σ_s w_s·L_s）

示例（bottleneck 恒为 1.0，扫 encoder3；再加一条四尺度的）：
  python sweep_feat.py --data_path /mnt2/songyd/5x5 \
    --arms "0.05:0,0,0.5,1.0" "0.05:0,0,0.9,1.0" "0.05:0,0,1.0,1.0" "0.05:0,0,1.5,1.0" \
           "0.05:0.1,0.2,0.5,1.0"

其余训练超参用与 baseline 一致的默认值（level4 / rtv=0 / 无 projector），
保证各臂之间**只有特征权重这一个变量**。已跑完的臂会自动跳过（除非 --force）。

每条臂训完自动做「多帧配对」评测（默认 5x5x4/0 前 50 帧 vs 同一 reference，降方差），
产出：summary.md 数值表（PSNR/ΔPSNR 的 mean±std、赢几帧）+ 三张图
（overview_psnr 各臂概览、dpsnr 配对Δ、curves_psnr 逐帧曲线）。给 --baseline_checkpoint
即与 N2N 同帧配对，判据为「ΔPSNR 的 mean > std 才算稳定赢」。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

SCALES = ["encoder1", "encoder2", "encoder3", "bottleneck"]   # 浅→深，须与 train_robust 的返回顺序一致
SHORT = {"encoder1": "e1", "encoder2": "e2", "encoder3": "e3", "bottleneck": "bn"}
CHANNELS = {"encoder1": 16, "encoder2": 32, "encoder3": 64, "bottleneck": 80}


def parse_arm(spec: str):
    """'[γ@]w_feat:w_enc1,w_enc2,w_enc3,w_bottleneck' -> (gamma|None, w_feat, [(scale,w),...])。
    可选前缀 'γ@' 逐臂覆盖一致性项 γ·Charb(f(n1),f(n2)) 的权重；不写则用全局 --gamma。
    尺度权重为 0 的尺度剔除。"""
    gamma = None
    if "@" in spec:
        g, spec = spec.split("@", 1)
        gamma = float(g)
    try:
        head, tail = spec.split(":")
        w_feat = float(head)
        ws = [float(x) for x in tail.split(",")]
    except ValueError:
        raise ValueError(f"臂格式错误: {spec!r}，应为 '[γ@]w_feat:w_enc1,w_enc2,w_enc3,w_bottleneck'")
    if len(ws) != len(SCALES):
        raise ValueError(f"臂 {spec!r} 需要 {len(SCALES)} 个尺度权重（enc1,enc2,enc3,bottleneck）")
    sel = [(s, w) for s, w in zip(SCALES, ws) if w > 0]
    if not sel:
        raise ValueError(f"臂 {spec!r} 所有尺度权重都是 0，没有特征损失可算")
    return gamma, w_feat, sel


def arm_name(gamma, w_feat, sel) -> str:
    parts = "_".join(f"{SHORT[s]}{w:g}" for s, w in sel)
    g = f"g{gamma:g}_" if gamma is not None else ""
    return f"{g}wf{w_feat:g}_{parts}"


def std_target(scale: str, args) -> float:
    """std 健康值 ≈ 1/√(z 的通道数)。无 projector 时 z=原生通道；有则为 feat_dim（0=原生）。"""
    dim = CHANNELS[scale] if (not args.feat_use_proj or args.feat_dim <= 0) else args.feat_dim
    return dim ** -0.5


def load_frames(scene_dir: str, n: int):
    """场景目录下前 n 帧 .npy（按文件名自然排序）。"""
    from pathlib import Path
    from eval_curve import natural_key
    frames = sorted(Path(scene_dir).glob("*.npy"), key=natural_key)[:n]
    if not frames:
        raise FileNotFoundError(f"{scene_dir} 下没找到 .npy 帧")
    return frames


def multiframe_eval(ckpt: str, frames, ref, dr, device):
    """对 ckpt 在 frames 上逐帧评测，返回 (psnr[], mssim[], r[]) 三个 numpy 数组。"""
    from eval_curve import run_curve, load_model
    import torch
    model = load_model(ckpt, device)
    p, s, r = run_curve(model, frames, ref, dr, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p, s, r


def mstd(a):
    """(mean, 样本std ddof=1)。"""
    import numpy as np
    return float(np.mean(a)), float(np.std(a, ddof=1) if len(a) > 1 else 0.0)


def parse_log(path: str):
    """取日志里最后一行 [EPOCH n] ...，解析成 {metric: value}。"""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("[EPOCH "):
                last = line
    if last is None:
        return None
    body = last.split("  saved=")[0]                       # 去掉尾部路径，免得被当成 k=v
    return {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?)", body)}


def main():
    p = argparse.ArgumentParser(description="特征一致性权重 sweep + 结果汇总")
    p.add_argument("--arms", type=str, nargs="+", required=True, help="每条臂: w_feat:w_enc1,w_enc2,w_enc3,w_bottleneck")
    # ---- 各臂共用的固定配置（保证唯一变量是特征权重）----
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--levels", type=int, nargs="*", default=[4])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--rtv_weight", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0, help="N2N 正向权重")
    p.add_argument("--beta", type=float, default=1.0, help="N2N 反向权重；0=单向（干净 base，等价原始 N2N）")
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--w_white", type=float, default=0.0)
    p.add_argument("--feat_use_proj", type=int, default=0)
    p.add_argument("--feat_dim", type=int, default=128, help="feat_use_proj=1 时生效；0=原生通道")
    p.add_argument("--feat_normalize", type=int, default=0, help="1=尺度权重归一化为和=1")
    p.add_argument("--feat_pool", type=int, default=0, help="投影前池化到 G×G（0=逐像素）")
    p.add_argument("--seed", type=int, default=42)
    # ---- 每条臂训完自动跑「多帧配对」评测（同一 reference，降方差）----
    p.add_argument("--eval", type=int, default=1, help="1=每条臂训完自动多帧评测并收指标")
    p.add_argument("--reference", type=str, default="/home/songyd/Projects/Robust-N2N/reference.npy")
    p.add_argument("--scene_dir", type=str, default="/mnt2/songyd/5x5/5x5x4/0/npy", help="评测帧目录（内含 .npy）")
    p.add_argument("--n_frames", type=int, default=50)
    p.add_argument("--baseline_checkpoint", type=str, default="", help="外部基线 checkpoint；给了则同帧配对对比")
    p.add_argument("--baseline_arm", type=str, default="",
                   help="用**本 sweep 内同 seed 的某条臂**当配对基线（如 '0:0,0,1,1' 即 w_feat=0）。"
                        "代码路径完全一致，消除 train_n2n/train_robust 的脚本差异。优先级高于 --baseline_checkpoint。")
    p.add_argument("--max", type=float, default=255.0, help="PSNR/MSSIM 的 data_range")
    p.add_argument("--device", type=str, default="")
    # ---- sweep 本身 ----
    p.add_argument("--root", type=str, default="results/sweep")
    p.add_argument("--force", action="store_true", help="已有 checkpoint 也重跑")
    p.add_argument("--dry_run", action="store_true", help="只打印命令，不实际训练")
    args = p.parse_args()

    arms = [parse_arm(a) for a in args.arms]
    ck_root, log_root = os.path.join(args.root, "checkpoints"), os.path.join(args.root, "logs")
    os.makedirs(ck_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(os.path.join(args.root, "eval"), exist_ok=True)

    # ---- 第一遍：只训练。基线臂也在其中，训完才能拿来配对 ----
    ckpt_of = {}                                            # arm name -> checkpoint 路径
    results = []
    for i, (gamma, w_feat, sel) in enumerate(arms, 1):
        g_val = args.gamma if gamma is None else gamma       # 逐臂 γ，未指定用全局
        name = arm_name(gamma, w_feat, sel)
        save_dir = os.path.join(ck_root, name)
        log_path = os.path.join(log_root, f"{name}.log")
        ckpt = os.path.join(save_dir, f"model_epoch_{args.epochs}.pth")

        cmd = [sys.executable, "train_robust.py",
               "--data_path", args.data_path,
               "--levels", *[str(x) for x in args.levels],
               "--epochs", str(args.epochs), "--crop_size", str(args.crop_size),
               "--batch_size", str(args.batch_size), "--lr", str(args.lr),
               "--rtv_weight", str(args.rtv_weight),
               "--alpha", str(args.alpha), "--beta", str(args.beta), "--gamma", str(g_val),
               "--w_white", str(args.w_white), "--seed", str(args.seed),
               "--w_feat", str(w_feat),
               "--feat_use_proj", str(args.feat_use_proj), "--feat_dim", str(args.feat_dim),
               "--feat_normalize", str(args.feat_normalize),
               "--feat_pool", str(args.feat_pool),
               "--feat_scales", *[s for s, _ in sel],
               "--feat_weights", *[str(w) for _, w in sel],
               "--save_dir", save_dir, "--log_dir", os.path.join(args.root, "tb", name)]

        print(f"\n=== [{i}/{len(arms)}] {name}  (gamma={g_val:g}) ===")
        if args.dry_run:
            print(" ".join(cmd)); continue
        if os.path.exists(ckpt) and not args.force:
            print(f"跳过（已有 {ckpt}）")
        else:
            with open(log_path, "w") as lf:
                ret = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            if ret.returncode != 0:
                print(f"[FAIL] 退出码 {ret.returncode}，看 {log_path}")
                results.append((name, g_val, w_feat, sel, None, None)); continue

        ckpt_of[name] = ckpt
        results.append((name, g_val, w_feat, sel, parse_log(log_path), None))

    if args.dry_run:
        return

    # ---- 第二遍：评测。基线可以是本 sweep 内同 seed 的某条臂（代码路径完全一致）----
    base_curve = frames = ref = device = dr = None
    if args.eval:
        import numpy as np
        import torch
        from infer_eval_robust import load2d
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        dr = float(args.max)
        ref = load2d(args.reference)
        frames = load_frames(args.scene_dir, args.n_frames)
        print(f"\n[INFO] eval: 场景={args.scene_dir} 前 {len(frames)} 帧; ref={args.reference} shape={ref.shape}; dr={dr:g}")

        base_ckpt, base_tag = "", ""
        if args.baseline_arm:                                # 优先：本 sweep 内的臂（同 seed、同代码路径）
            bg, bwf, bsel = parse_arm(args.baseline_arm)
            bname = arm_name(bg, bwf, bsel)
            if bname not in ckpt_of or not os.path.exists(ckpt_of[bname]):
                raise RuntimeError(f"--baseline_arm 指定的臂 {bname} 不在本 sweep 里或未训练成功")
            base_ckpt, base_tag = ckpt_of[bname], f"同 seed 臂 {bname}"
        elif args.baseline_checkpoint:
            base_ckpt, base_tag = args.baseline_checkpoint, os.path.basename(args.baseline_checkpoint)
        if base_ckpt:
            bp, bs, br = multiframe_eval(base_ckpt, frames, ref, dr, device)
            base_curve = (bp, bs, br)
            print(f"[INFO] 基线（{base_tag}）: PSNR {np.mean(bp):.3f}±{np.std(bp, ddof=1):.3f}  "
                  f"MSSIM {np.mean(bs):.4f}  r {np.mean(br):.4f}")

        for k, (name, g_val, w_feat, sel, m, _) in enumerate(results):
            ckpt = ckpt_of.get(name, "")
            if m is None or not ckpt or not os.path.exists(ckpt):
                continue
            p_arr, s_arr, r_arr = multiframe_eval(ckpt, frames, ref, dr, device)
            results[k] = (name, g_val, w_feat, sel, m, {"p": p_arr, "s": s_arr, "r": r_arr})
            pm, ps = mstd(p_arr)
            msg = f"  {name:<26} PSNR {pm:.3f}±{ps:.3f}  MSSIM {np.mean(s_arr):.4f}  r {np.mean(r_arr):.4f}"
            if base_curve is not None:
                d = p_arr - base_curve[0]
                dm, ds = mstd(d)
                msg += f"  | ΔPSNR {dm:+.3f}±{ds:.3f}  赢 {int((d > 0).sum())}/{len(d)}"
            print(msg)
            with open(os.path.join(args.root, "eval", f"{name}.csv"), "w") as f:
                base_p = base_curve[0] if base_curve is not None else [None] * len(frames)
                f.write("frame,psnr,mssim,r" + (",base_psnr,dPSNR\n" if base_curve is not None else "\n"))
                for j, fp in enumerate(frames):
                    row = f"{fp.stem},{p_arr[j]:.4f},{s_arr[j]:.4f},{r_arr[j]:.4f}"
                    if base_curve is not None:
                        row += f",{base_p[j]:.4f},{p_arr[j] - base_p[j]:.4f}"
                    f.write(row + "\n")

    build_report(results, base_curve, frames, args)


def build_report(results, base_curve, frames, args):
    """汇总：数值表格(summary.md) + 曲线图(概览/配对Δ/逐帧)。"""
    import numpy as np
    base_p = base_curve[0] if base_curve is not None else None
    base_pm, base_ps = mstd(base_p) if base_p is not None else (None, None)

    lines = ["# 特征一致性权重 sweep 汇总（多帧平均，降方差）", "",
             f"共用配置: levels={args.levels} epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
             f"rtv={args.rtv_weight} alpha={args.alpha} beta={args.beta} gamma={args.gamma} feat_use_proj={args.feat_use_proj} "
             f"feat_normalize={args.feat_normalize} seed={args.seed}",
             f"评测: {args.scene_dir} 前 {len(frames) if frames else 0} 帧 vs {os.path.basename(args.reference)} "
             f"(data_range={args.max:g})", ""]
    if base_p is not None:
        lines += [f"**基线**（{args.baseline_arm or os.path.basename(args.baseline_checkpoint)}）: "
                  f"PSNR {base_pm:.3f}±{base_ps:.3f}  MSSIM {np.mean(base_curve[1]):.4f}  r {np.mean(base_curve[2]):.4f}", ""]
    lines += ["判据：**ΔPSNR 的 mean 要大于 std 才算稳定赢过 N2N**（否则差异在噪声内）。"
              "`std[·]` 括号内为塌缩健康值≈1/√dim。", "",
              "| 臂 | γ | w_feat | 尺度(权重) | PSNR mean±std | ΔPSNR mean±std | 赢/N | MSSIM | r | rec | feat | "
              + " | ".join(f"std[{SHORT[s]}]" for s in SCALES) + " |",
              "|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(SCALES)]

    plot_rows = []                                            # (name, p_arr) 供画图
    for name, g_val, w_feat, sel, m, ev in results:
        scales_txt = ", ".join(f"{SHORT[s]}={w:g}" for s, w in sel)
        if m is None or ev is None:
            lines.append(f"| {name} | {g_val:g} | {w_feat:g} | {scales_txt} | 失败/未评测 |" + " - |" * (6 + len(SCALES))); continue
        pm, ps = mstd(ev["p"])
        if base_p is not None:
            d = ev["p"] - base_p
            dm, ds = mstd(d)
            dpsnr = f"**{dm:+.3f}**±{ds:.3f}" if dm > ds else f"{dm:+.3f}±{ds:.3f}"   # 稳定赢的加粗
            win = f"{int((d > 0).sum())}/{len(d)}"
        else:
            dpsnr, win = "-", "-"
        std_by_scale = {s: m.get(f"std{i}") for i, (s, _) in enumerate(sel)}
        cells = ["-" if std_by_scale.get(s) is None else f"{std_by_scale[s]:.3f} ({std_target(s, args):.3f})" for s in SCALES]
        lines.append(f"| {name} | {g_val:g} | {w_feat:g} | {scales_txt} | {pm:.3f}±{ps:.3f} | {dpsnr} | {win} | "
                     f"{np.mean(ev['s']):.4f} | {np.mean(ev['r']):.4f} | {m.get('rec', float('nan')):.5f} | "
                     f"{m.get('feat', float('nan')):.4f} | " + " | ".join(cells) + " |")
        plot_rows.append((name, ev["p"]))

    out = os.path.join(args.root, "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    make_plots(plot_rows, base_curve, args)
    print(f"\n[OK] 汇总表 -> {out}")
    print(f"[OK] 曲线图 -> {args.root}/overview_psnr.png"
          + ("" if base_curve is None else f", dpsnr.png") + f", curves_psnr.png")


def make_plots(plot_rows, base_curve, args):
    """画三张图：各臂 PSNR 概览(mean±std)、配对 ΔPSNR、逐帧曲线叠加。"""
    if not plot_rows:
        return
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _ in plot_rows]
    means = np.array([np.mean(p) for _, p in plot_rows])
    stds = np.array([np.std(p, ddof=1) if len(p) > 1 else 0.0 for _, p in plot_rows])
    x = np.arange(len(names))

    # 图1：各臂 PSNR mean±std（说服力最强的一张）
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 5), dpi=140)
    ax.errorbar(x, means, yerr=stds, fmt="o", ms=5, capsize=4, color="#D4537E", label="Robust-N2N arms")
    if base_curve is not None:
        bpm = float(np.mean(base_curve[0])); bps = float(np.std(base_curve[0], ddof=1))
        ax.axhline(bpm, ls="--", color="#378ADD", label=f"N2N baseline ({bpm:.3f})")
        ax.axhspan(bpm - bps, bpm + bps, color="#378ADD", alpha=0.12)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PSNR (dB)  mean ± std over frames"); ax.set_title("各臂多帧 PSNR 概览")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.root, "overview_psnr.png"), bbox_inches="tight"); plt.close(fig)

    # 图2：配对 ΔPSNR mean±std（>0 且超出 std 才算赢）
    if base_curve is not None:
        dm = np.array([np.mean(p - base_curve[0]) for _, p in plot_rows])
        ds = np.array([np.std(p - base_curve[0], ddof=1) if len(p) > 1 else 0.0 for _, p in plot_rows])
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 5), dpi=140)
        colors = ["#1D9E75" if m > s else "#B4B2A9" for m, s in zip(dm, ds)]
        ax.bar(x, dm, yerr=ds, capsize=4, color=colors)
        ax.axhline(0, color="#444441", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("ΔPSNR vs N2N (dB)"); ax.set_title("配对 ΔPSNR（绿=mean>std 稳定赢；灰=噪声内）")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(args.root, "dpsnr.png"), bbox_inches="tight"); plt.close(fig)

    # 图3：逐帧 PSNR 曲线叠加
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    fx = np.arange(len(plot_rows[0][1]))
    for n, p in plot_rows:
        ax.plot(fx, p, "-", lw=1, alpha=0.8, label=n)
    if base_curve is not None:
        ax.plot(fx, base_curve[0], "-k", lw=2, label="N2N baseline")
    ax.set_xlabel("frame index"); ax.set_ylabel("PSNR (dB)"); ax.set_title("逐帧 PSNR 曲线")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(args.root, "curves_psnr.png"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
