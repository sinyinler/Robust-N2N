# -*- coding: utf-8 -*-
"""诊断 #1：native 编码器特征是否存在真实的表征塌缩。

为什么要换指标：训练时用的 `F.normalize(z,dim=1).std(dim=(0,2,3)).mean()` 有两处缺陷——
  (a) 把「样本间差异」与「空间纹理差异」混在一起：一个对所有输入输出同一张空间花纹的解，
      也会因 H/W 上有变化而显示 std 健康；
  (b) 它测的是 projector 输出 z，而 z 的末层是 BatchNorm —— BN 会把逐通道方差顶起来。
      因此「无 projector(z=native, 无BN) vs 带 projector(z 有BN)」的 std 根本不可比。

本脚本改测 **decoder 真正使用的 native out3 / bridge**，并给出三类互补指标：
  1) 跨样本 std：先沿 H,W 池化 → (N,C)，再沿 N 求逐通道 std。只有它能检出「常量塌缩」。
  2) 空间内 std：沿 H,W 求 std 再对 N,C 平均。与 (1) 对照可分离「空间花纹」与「样本响应」。
  3) effective rank（RankMe, Garrido et al. ICML 2023）+ 协方差谱：检出维度塌缩/通道冗余，
     这是 std 无论如何都看不出来的。RankMe = exp(-Σ p_k log p_k), p_k = σ_k / Σσ。

注：训练时 projector/predictor 存在 criterion_feat 里、不进 checkpoint，故无法事后分析 z。
若需分析 z，请用 train_robust.py 的 --save_feat_head 1 重新训练。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diag_common import collect_frames, load_batch
from models.denoiser_feats import DenoiserWithFeats
from utils.checkpoint import load_weights_flexible


def effective_rank(mat: torch.Tensor, eps: float = 1e-12) -> float:
    """RankMe：对 (M,C) 矩阵取奇异值熵的指数。M 为样本数，C 为通道数。"""
    sv = torch.linalg.svdvals(mat.double())
    p = sv / (sv.sum() + eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()))


def spectrum_stats(mat: torch.Tensor):
    """协方差谱：返回 (top1 占比, top10% 占比, 参与比 PR)。"""
    x = mat - mat.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp_min(0).flip(0)   # 降序
    tot = ev.sum() + 1e-12
    k = max(1, int(0.1 * ev.numel()))
    pr = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))          # participation ratio
    return float(ev[0] / tot), float(ev[:k].sum() / tot), pr


def report(name: str, feat: torch.Tensor, max_pix: int = 20000):
    """feat: (N,C,H,W) —— 打印三类指标。"""
    n, c, h, w = feat.shape
    pooled = feat.mean(dim=(2, 3))                                   # (N,C) 空间池化
    cross_sample_std = pooled.std(dim=0, unbiased=False).mean().item() if n > 1 else float("nan")
    within_spatial_std = feat.std(dim=(2, 3), unbiased=False).mean().item()

    # 逐像素特征（子采样）用于维度塌缩分析
    flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
    if flat.shape[0] > max_pix:
        sel = torch.randperm(flat.shape[0], device=flat.device)[:max_pix]
        flat = flat[sel]
    er = effective_rank(flat)
    top1, top10, pr = spectrum_stats(flat)

    # 旧监控（作对照，说明它为何不可靠）
    old = F.normalize(feat, dim=1).std(dim=(0, 2, 3)).mean().item()

    print(f"\n  [{name}]  shape=(N={n}, C={c}, H={h}, W={w})")
    print(f"    跨样本 std（池化后沿 N）      : {cross_sample_std:.5f}   ← 趋 0 = 常量塌缩")
    print(f"    空间内 std（沿 H,W）          : {within_spatial_std:.5f}   ← 与上行对照，分离空间花纹")
    print(f"    effective rank (RankMe)      : {er:.2f} / {c}   ({er / c * 100:.1f}% of C)")
    print(f"    协方差谱 top1 / top10% 占比   : {top1 * 100:.1f}% / {top10 * 100:.1f}%   参与比 PR={pr:.2f}")
    print(f"    [旧监控 normalize.std(0,2,3)] : {old:.5f}   （1/√C={c ** -0.5:.4f}）← 混淆量，仅作对照")


@torch.no_grad()
def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    frames, multi_scene = collect_frames(args.scene_root, args.n_scenes, args.frame_idx,
                                         args.scene_dir, args.batch)
    if not multi_scene:
        print("[WARN] 同场景 batch：「跨样本 std」不可解读（同场景不同帧的深层特征本就该相似）。"
              "\n       effective rank / 协方差谱不受影响，仍然有效。")

    batch = load_batch(frames, args.crop, device)    # 中心裁剪到统一尺寸（不同场景尺寸不同）
    print(f"[INFO] batch shape = {tuple(batch.shape)}")

    for tag, ckpt in [("N2N", args.n2n_checkpoint), ("+feature", args.feat_checkpoint)]:
        if not ckpt:
            continue
        model = DenoiserWithFeats(input_channels=1).to(device).eval()
        print(f"\n=== {tag} === {ckpt}", load_weights_flexible(model, ckpt, device))
        _, feats = model(batch, return_feats=True)      # [enc1, enc2, enc3, bottleneck]
        for name, f in zip(["encoder1", "encoder2", "encoder3(out3)", "bottleneck(bridge)"], feats):
            if name.startswith("encoder1") and not args.all_scales:
                continue
            if name.startswith("encoder2") and not args.all_scales:
                continue
            report(name, f.float())

    print("\n判读：跨样本 std → 0 才是常量塌缩；std 正常但 effective rank 远小于 C、"
          "\n      或 top10% 主成分吃掉绝大部分方差 → 维度塌缩/通道冗余（旧监控看不出来）。")


def parse_args():
    p = argparse.ArgumentParser(description="native 编码器特征的表征健康度诊断（跨样本 std + 有效秩 + 协方差谱）")
    p.add_argument("--n2n_checkpoint", default="")
    p.add_argument("--feat_checkpoint", default="")
    p.add_argument("--scene_root", default="/mnt2/songyd/5x5/5x5x4", help="其下每个子目录是一个场景（推荐）")
    p.add_argument("--n_scenes", type=int, default=8)
    p.add_argument("--frame_idx", type=int, default=0)
    p.add_argument("--scene_dir", default="", help="退回同场景模式（跨样本 std 不可解读）")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--crop", type=int, default=512, help="中心裁剪到该尺寸")
    p.add_argument("--all_scales", type=int, default=0, help="1=连 encoder1/2 一起报")
    p.add_argument("--device", default="")
    a = p.parse_args()
    if a.scene_dir:
        a.scene_root = ""
    return a


if __name__ == "__main__":
    main(parse_args())
