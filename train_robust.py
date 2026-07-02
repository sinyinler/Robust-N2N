# -*- coding: utf-8 -*-
"""Robust-N2N 一步法训练入口。

数据与配对完全复用 train_n2n 的 build_loaders（同场景 Δ∈{5,7,9} 的噪声对 n1/n2，
支持 /mnt2/songyd/mix 的 npy 与 lbf）。区别：
  - 网络：RobustDenoiser（轻量 U-Net + 3 处 GIBlock，训练注入/推理关闭）；
  - 每个 batch 对 n1、n2 都前向，得到 f(n1)、f(n2)；
  - 损失：RobustN2NLoss = 对称N2N(Charbonnier) + 一致性 + [过半后]残差白度 + RTV；
  - 白度项在训练进度过 whiten_start_frac 后启用。
"""
from __future__ import annotations

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 复用 N2N 的数据加载 / 调度 / 随机种子（数据与配对口径完全一致）
from train_n2n import build_loaders, set_seed, build_onecycle
from models.robust_denoiser import RobustDenoiser
from losses.robust_n2n_loss import RobustN2NLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robust-N2N one-step training (LiteN2N U-Net + GIBlock + combined loss).")
    # ---- 数据（与 train_n2n 一致，供 build_loaders 使用）----
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--data_subdir", type=str, default="npy")
    p.add_argument("--strict_data_subdir", type=int, default=1)
    p.add_argument("--data_index_min", type=int, default=-1)
    p.add_argument("--data_index_max", type=int, default=-1)
    p.add_argument("--levels", type=int, nargs="*", default=None)
    p.add_argument("--mix_root", type=str, default="")
    p.add_argument("--mix_scenes", type=str, nargs="*", default=None)
    p.add_argument("--mix_subdirs", type=str, nargs="*", default=None)
    p.add_argument("--intervals", type=int, nargs="*", default=[5, 7, 9])
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=24, help="对 n1、n2 各前向一次，显存约 2×，故默认比 N2N 小。")
    p.add_argument("--max_pixels_per_batch", type=int, default=0)
    p.add_argument("--batch_ref_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--train_fraction", type=float, default=0.99)
    p.add_argument("--val_limit_batches", type=int, default=20)
    p.add_argument("--intensity_transform", choices=["log1p", "boxcox", "learned_vst"], default="log1p")
    p.add_argument("--vst_lut", type=str, default="")
    p.add_argument("--boxcox_lam", type=float, default=-0.15)
    p.add_argument("--boxcox_eps", type=float, default=1e-6)
    p.add_argument("--lambda_conditioned", type=int, default=0)
    p.add_argument("--lambda_min", type=float, default=-0.3)
    p.add_argument("--lambda_max", type=float, default=0.2)
    p.add_argument("--lambda_candidates", type=float, nargs="*",
                   default=[-0.3, -0.25, -0.2, -0.15, -0.1, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2])
    # ---- 训练调度 ----
    p.add_argument("--save_dir", type=str, default="results/checkpoints/robust_n2n")
    p.add_argument("--log_dir", type=str, default="results/logs/robust_n2n")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01, help="alias for lr_max")
    p.add_argument("--lr_max", type=float, default=None)
    p.add_argument("--lr_final", type=float, default=0.0005)
    p.add_argument("--warmup_pct", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--data_parallel", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    # ---- 组合损失 / GIBlock ----
    p.add_argument("--alpha", type=float, default=1.0, help="N2N 正向 Charbonnier 权重")
    p.add_argument("--beta", type=float, default=1.0, help="N2N 反向 Charbonnier 权重")
    p.add_argument("--gamma", type=float, default=0.1, help="一致性 |f(n1)-f(n2)| 权重")
    p.add_argument("--w_white", type=float, default=0.05, help="残差白度总权重（小）")
    p.add_argument("--beta_freq", type=float, default=2e-3, help="白度内部 freq 相对 spatial 的权重")
    p.add_argument("--rtv_weight", type=float, default=0.01, help="RTV 权重（建议对齐原 N2N 训练）")
    p.add_argument("--highpass_ratio", type=float, default=0.0)
    p.add_argument("--whiten_start_frac", type=float, default=0.5, help="训练进度过此比例后才开白度项")
    p.add_argument("--inject_sigma", type=float, default=1.0, help="GIBlock 注入高斯基准强度（推理时关闭）")
    p.add_argument("--init_noise_scale", type=float, default=0.1, help="GIBlock 注入可学习强度初值")
    args = p.parse_args()
    if args.data_index_min < 0:
        args.data_index_min = None
    if args.data_index_max < 0:
        args.data_index_max = None
    if args.lr_max is None:
        args.lr_max = args.lr
    if args.lr_max <= args.lr_final:
        raise ValueError(f"lr_max({args.lr_max}) must be > lr_final({args.lr_final})")
    return args


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.save_dir, exist_ok=True)

    _, train_loader, val_loader = build_loaders(args)
    model = RobustDenoiser(input_channels=1, inject_sigma=args.inject_sigma,
                           init_noise_scale=args.init_noise_scale).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    criterion = RobustN2NLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                              w_white=args.w_white, beta_freq=args.beta_freq,
                              w_rtv=args.rtv_weight, highpass_ratio=args.highpass_ratio).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=1e-4)
    scheduler = build_onecycle(optimizer, len(train_loader), args)

    total_steps = args.epochs * len(train_loader)
    whiten_start = int(args.whiten_start_frac * total_steps)
    print(f"[INFO] total_steps={total_steps}, whitening starts at step {whiten_start} "
          f"(frac={args.whiten_start_frac}); train batches/epoch={len(train_loader)}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()  # GIBlock 注入在训练模式下生效
        running = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for n1, n2 in pbar:
            n1 = n1.to(device, non_blocking=True)
            n2 = n2.to(device, non_blocking=True)
            f_n1 = model(n1)
            f_n2 = model(n2)
            use_white = global_step >= whiten_start
            loss, logs = criterion(f_n1, n1, f_n2, n2, use_whitening=use_white)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            pbar.set_postfix({"loss": f"{logs['total']:.5f}", "white": "on" if use_white else "off",
                              "lr": f"{scheduler.get_last_lr()[0]:.2g}"})

        n = max(1, len(train_loader))
        avg = {k: v / n for k, v in running.items()}
        save_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pth")
        state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(state, save_path)
        print(f"[EPOCH {epoch}] " + " ".join(f"{k}={avg[k]:.5f}" for k in
              ("total", "rec", "cons", "rtv", "white", "spatial", "freq") if k in avg)
              + f"  saved={save_path}")


if __name__ == "__main__":
    train(parse_args())
