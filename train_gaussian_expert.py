from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.ntn_dataset import N2NBootstrapTripletDataset, mix_sources_from_args
from losses.charbonnier import CharbonnierLoss
from models.denoiser import Denoiser
from utils.checkpoint import load_weights_flexible, save_training_checkpoint
from utils.intensity import append_condition_channel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def condition_or_none(batch: dict, device: torch.device) -> torch.Tensor | None:
    condition = batch.get("condition")
    if condition is None or condition.numel() == 0:
        return None
    return condition.to(device)


def build_dataset(args) -> N2NBootstrapTripletDataset:
    return N2NBootstrapTripletDataset(
        root_dir=args.data_path,
        intervals=args.intervals,
        crop_size=args.crop_size,
        random_crop=bool(args.random_crop),
        pseudo_clean_frames=args.pseudo_clean_frames,
        data_subdirs=tuple(args.data_subdirs),
        strict_data_subdir=bool(args.strict_data_subdir),
        data_index_min=args.data_index_min if args.data_index_min >= 0 else None,
        data_index_max=args.data_index_max if args.data_index_max >= 0 else None,
        include_levels=tuple(args.levels) if args.levels else None,
        extra_sources=mix_sources_from_args(args),
        intensity_transform=args.intensity_transform,
        boxcox_lam=args.boxcox_lam,
        boxcox_eps=args.boxcox_eps,
        lambda_conditioned=bool(args.lambda_conditioned),
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_candidates=args.lambda_candidates,
        vst_lut=args.vst_lut,
        augment=True,
        compute_pseudo_clean=not bool(args.bootstrap_checkpoint),
    )


def maybe_build_bootstrap(args, device: torch.device, input_channels: int) -> Denoiser | None:
    if not args.bootstrap_checkpoint:
        return None
    model = Denoiser(input_channels=input_channels).to(device)
    info = load_weights_flexible(model, args.bootstrap_checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[INFO] Loaded N2N bootstrap checkpoint: {info}")
    return model


def maybe_data_parallel(model: torch.nn.Module, args, name: str) -> torch.nn.Module:
    """和原 N2N 训练保持一致：多卡可用时默认启用 DataParallel。"""

    if torch.cuda.device_count() > 1 and bool(args.data_parallel):
        print(f"[INFO] Using DataParallel for {name} on {torch.cuda.device_count()} GPUs")
        return torch.nn.DataParallel(model)
    return model


def build_onecycle(optimizer, steps_per_epoch: int, args):
    total_steps = max(1, steps_per_epoch * args.epochs)
    return OneCycleLR(
        optimizer,
        max_lr=args.lr_max,
        total_steps=total_steps,
        pct_start=args.warmup_pct,
        anneal_strategy="cos",
        div_factor=args.lr_max / args.lr_final,
        final_div_factor=1.0,
        cycle_momentum=False,
    )


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    val_size = max(1, int(len(dataset) * args.val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    input_channels = 2 if args.lambda_conditioned and args.intensity_transform == "boxcox" else 1
    model = Denoiser(input_channels=input_channels).to(device)
    bootstrap_model = maybe_build_bootstrap(args, device, input_channels=input_channels)
    model = maybe_data_parallel(model, args, name="Gaussian expert D_prime")
    if bootstrap_model is not None:
        bootstrap_model = maybe_data_parallel(bootstrap_model, args, name="frozen N2N bootstrap")
    criterion = CharbonnierLoss(eps=args.charbonnier_eps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_final,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = build_onecycle(optimizer, len(train_loader), args)

    metadata = vars(args).copy()
    metadata.update({"dataset_size": len(dataset), "train_size": train_size, "val_size": val_size})
    (save_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        pbar = tqdm(train_loader, desc=f"D' epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            image = batch["input"].to(device)
            pseudo_clean = batch["pseudo_clean"].to(device)
            condition = condition_or_none(batch, device)

            if bootstrap_model is not None:
                with torch.no_grad():
                    pseudo_clean = bootstrap_model(append_condition_channel(image, condition)).detach()

            sigma = torch.empty(image.shape[0], 1, 1, 1, device=device).uniform_(args.sigma_min, args.sigma_max)
            noisy = pseudo_clean + torch.randn_like(pseudo_clean) * sigma
            pred = model(append_condition_channel(noisy, condition))
            loss = criterion(pred, pseudo_clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.item())
            total += loss_value
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                {
                    "loss": f"{loss_value:.6f}",
                    "avg": f"{total / step:.6f}",
                    "lr": f"{current_lr:.3g}",
                }
            )

        train_loss = total / max(1, len(train_loader))
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    image = batch["input"].to(device)
                    pseudo_clean = batch["pseudo_clean"].to(device)
                    condition = condition_or_none(batch, device)
                    if bootstrap_model is not None:
                        pseudo_clean = bootstrap_model(append_condition_channel(image, condition)).detach()
                    sigma = torch.empty(image.shape[0], 1, 1, 1, device=device).uniform_(args.sigma_min, args.sigma_max)
                    noisy = pseudo_clean + torch.randn_like(pseudo_clean) * sigma
                    pred = model(append_condition_channel(noisy, condition))
                    val_loss += float(criterion(pred, pseudo_clean).item())
            val_loss /= max(1, len(val_loader))

        print(f"[EPOCH {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        save_training_checkpoint(save_dir / f"gaussian_expert_epoch_{epoch}.pth", model, epoch, args, extra={"train_loss": train_loss, "val_loss": val_loss})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: train Gaussian expert D' with pseudo-clean bootstrap targets.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_subdirs", nargs="*", default=["npy", "lbf"])
    parser.add_argument("--strict_data_subdir", type=int, default=0)
    parser.add_argument("--data_index_min", type=int, default=-1)
    parser.add_argument("--data_index_max", type=int, default=-1)
    parser.add_argument("--levels", type=int, nargs="*", default=None,
                        help="只用这些叠加层级 5x5xN 的 N（如 --levels 2 3 4 把 level1 留作 OOD 测试）。")
    parser.add_argument("--mix_root", type=str, default="",
                        help="额外数据根目录（如 /mnt2/songyd/mix），配合 --mix_scenes 加入多被试训练。")
    parser.add_argument("--mix_scenes", type=str, nargs="*", default=None,
                        help="只取 mix_root 下这些场景编号（如脑 305..312 + 腿 316..321；手 325 留作 OOD 不要列入）。")
    parser.add_argument("--mix_subdirs", type=str, nargs="*", default=None,
                        help="mix 数据子目录（默认同 --data_subdirs，会自动兼容直接 lbf 与 npy 子目录两种结构）。")
    parser.add_argument("--intervals", type=int, nargs="*", default=[5, 7, 9])
    # D' 与 N2N 同网络、同数据、同任务，crop/batch 对齐 N2N（512/48）：上下文一致、BN 统计稳、用满 GPU。
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--random_crop", type=int, default=1)
    parser.add_argument("--pseudo_clean_frames", type=int, default=0, help="0 means average all frames in the sequence.")
    parser.add_argument("--bootstrap_checkpoint", type=str, default="", help="Optional trained N2N model used as C_hat generator.")
    parser.add_argument("--intensity_transform", choices=["none", "log1p", "boxcox", "learned_vst"], default="log1p")
    parser.add_argument("--vst_lut", type=str, default="")
    parser.add_argument("--boxcox_lam", type=float, default=-0.15)
    parser.add_argument("--boxcox_eps", type=float, default=1e-6)
    parser.add_argument("--lambda_conditioned", type=int, default=0)
    parser.add_argument("--lambda_min", type=float, default=-0.3)
    parser.add_argument("--lambda_max", type=float, default=0.2)
    parser.add_argument("--lambda_candidates", type=float, nargs="*", default=None)
    # 盲高斯区间：覆盖全部训练被试的真实噪声跨度（log1p 域实测：腿≈0.03、脑≈0.07、
    # 5x5 level4≈0.10 ~ level1≈0.43、手≈0.37）。下界 0.02 盖住腿并留余量、上界 0.6 盖住 5x5 最噪。
    parser.add_argument("--sigma_min", type=float, default=0.02)
    parser.add_argument("--sigma_max", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.01, help="Compatibility alias for lr_max.")
    parser.add_argument("--lr_max", type=float, default=None)
    parser.add_argument("--lr_final", type=float, default=0.0005)
    parser.add_argument("--warmup_pct", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--data_parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="results/checkpoints/gaussian_expert")
    args = parser.parse_args()
    if args.lr_max is None:
        args.lr_max = args.lr
    if args.lr_max <= args.lr_final:
        raise ValueError(f"lr_max must be greater than lr_final, got lr_max={args.lr_max}, lr_final={args.lr_final}")
    return args


if __name__ == "__main__":
    train(parse_args())
