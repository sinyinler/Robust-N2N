# -*- coding: utf-8 -*-
"""跨视图特征一致性权重 sweep：一次铺完多条臂，跑完自动汇总成表。

臂的格式： w_feat:w_enc1,w_enc2,w_enc3,w_bottleneck
  - 权重为 0 的尺度**不参与**（不建 projector/predictor，省算力与显存）
  - 特征项实际总权重 = w_feat × 该尺度权重（代码里 L_feat = Σ_s w_s·L_s）

示例（bottleneck 恒为 1.0，扫 encoder3；再加一条四尺度的）：
  python sweep_feat.py --data_path /mnt2/songyd/5x5 \
    --arms "0.05:0,0,0.5,1.0" "0.05:0,0,0.9,1.0" "0.05:0,0,1.0,1.0" "0.05:0,0,1.5,1.0" \
           "0.05:0.1,0.2,0.5,1.0"

其余训练超参用与 baseline 一致的默认值（level4 / 1 epoch / rtv=0 / 无 projector），
保证各臂之间**只有特征权重这一个变量**。已跑完的臂会自动跳过（除非 --force）。
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


def parse_eval(out: str):
    """解析 infer_eval_robust.py 的指标表，取 Robust-N2N 与 N2N (ours) 两行。"""
    def row(tag):
        m = re.search(rf"^\s*{tag}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", out, re.M)
        return dict(zip(("psnr", "mssim", "r"), map(float, m.groups()))) if m else None
    return row(r"Robust-N2N"), row(r"N2N \(ours\)")


def run_eval(ckpt: str, name: str, args):
    """对该臂的 checkpoint 跑单图推理评测，返回 (robust指标, n2n基线指标)。"""
    out_dir = os.path.join(args.root, "infer", name)
    cmd = [sys.executable, "infer_eval_robust.py", "--checkpoint", ckpt,
           "--raw", args.raw, "--reference", args.reference, "--out_dir", out_dir]
    if args.n2n:
        cmd += ["--n2n", args.n2n]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(args.root, "logs", f"{name}.eval.log"), "w") as f:
        f.write(ret.stdout + ret.stderr)
    if ret.returncode != 0:
        print(f"[FAIL eval] 退出码 {ret.returncode}，看 {name}.eval.log")
        return None, None
    return parse_eval(ret.stdout)


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
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--w_white", type=float, default=0.0)
    p.add_argument("--feat_use_proj", type=int, default=0)
    p.add_argument("--feat_dim", type=int, default=128, help="feat_use_proj=1 时生效；0=原生通道")
    p.add_argument("--seed", type=int, default=42)
    # ---- 每条臂训完自动跑单图评测（infer_eval_robust.py）----
    p.add_argument("--eval", type=int, default=1, help="1=每条臂训完自动评测并收指标")
    p.add_argument("--raw", type=str, default="/home/songyd/Projects/Robust-N2N/raw.npy")
    p.add_argument("--reference", type=str, default="/home/songyd/Projects/Robust-N2N/reference.npy")
    p.add_argument("--n2n", type=str, default="/home/songyd/Projects/Robust-N2N/ours.npy", help="N2N 基线结果；空则不比")
    # ---- sweep 本身 ----
    p.add_argument("--root", type=str, default="results/sweep")
    p.add_argument("--force", action="store_true", help="已有 checkpoint 也重跑")
    p.add_argument("--dry_run", action="store_true", help="只打印命令，不实际训练")
    args = p.parse_args()

    arms = [parse_arm(a) for a in args.arms]
    ck_root, log_root = os.path.join(args.root, "checkpoints"), os.path.join(args.root, "logs")
    os.makedirs(ck_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    results, n2n_base = [], None
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
               "--rtv_weight", str(args.rtv_weight), "--gamma", str(g_val),
               "--w_white", str(args.w_white), "--seed", str(args.seed),
               "--w_feat", str(w_feat),
               "--feat_use_proj", str(args.feat_use_proj), "--feat_dim", str(args.feat_dim),
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

        ev = None
        if args.eval and os.path.exists(ckpt):
            ev, base = run_eval(ckpt, name, args)
            if base and n2n_base is None:
                n2n_base = base                          # N2N 基线各臂相同，记一次即可
            if ev:
                print(f"  PSNR={ev['psnr']:.3f}  MSSIM={ev['mssim']:.4f}  r={ev['r']:.4f}")
        results.append((name, g_val, w_feat, sel, parse_log(log_path), ev))

    if args.dry_run:
        return

    # ---- 汇总表 ----
    base_psnr = n2n_base["psnr"] if n2n_base else None
    lines = ["# 特征一致性权重 sweep 汇总", "",
             f"共用配置: levels={args.levels} epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
             f"rtv={args.rtv_weight} gamma={args.gamma} feat_use_proj={args.feat_use_proj} seed={args.seed}", ""]
    if n2n_base:
        lines += [f"**N2N 基线**（{os.path.basename(args.n2n)}）: PSNR={n2n_base['psnr']:.3f}  "
                  f"MSSIM={n2n_base['mssim']:.4f}  r={n2n_base['r']:.4f} —— ΔPSNR>0 才算赢过 N2N。", ""]
    lines += ["`std` 括号内为健康值 ≈1/√dim；往 0 掉 = 塌缩。`rec` 越低越好，`feat` 越负说明特征越对齐。"
              "`γ` = 一致性项 Charb(f(n1),f(n2)) 的权重。", "",
              "| 臂 | γ | w_feat | 尺度(权重) | PSNR | ΔPSNR | MSSIM | r | rec | diff | feat | "
              + " | ".join(f"std[{SHORT[s]}]" for s in SCALES) + " |",
              "|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(SCALES)]
    for name, g_val, w_feat, sel, m, ev in results:
        scales_txt = ", ".join(f"{SHORT[s]}={w:g}" for s, w in sel)
        if m is None:
            lines.append(f"| {name} | {g_val:g} | {w_feat:g} | {scales_txt} | 失败 |" + " - |" * (6 + len(SCALES))); continue
        if ev:
            d = f"{ev['psnr'] - base_psnr:+.3f}" if base_psnr else "-"
            ev_txt = f"{ev['psnr']:.3f} | {d} | {ev['mssim']:.4f} | {ev['r']:.4f}"
        else:
            ev_txt = "- | - | - | -"
        # std{i} 按 feat_scales 的顺序对应 sel[i]，映射回尺度名
        std_by_scale = {s: m.get(f"std{i}") for i, (s, _) in enumerate(sel)}
        cells = []
        for s in SCALES:
            v = std_by_scale.get(s)
            cells.append("-" if v is None else f"{v:.3f} ({std_target(s, args):.3f})")
        lines.append(f"| {name} | {g_val:g} | {w_feat:g} | {scales_txt} | {ev_txt} | {m.get('rec', float('nan')):.5f} | "
                     f"{m.get('diff', float('nan')):.5f} | {m.get('feat', float('nan')):.4f} | " + " | ".join(cells) + " |")

    out = os.path.join(args.root, "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n[OK] 汇总已写入 {out}")


if __name__ == "__main__":
    main()
