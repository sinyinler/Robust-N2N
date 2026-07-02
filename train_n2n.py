from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print("TensorBoard is not installed; scalar logging is disabled.")

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from data.legacy_pairs import SpeckleN2NLogDataset
from data.ntn_dataset import N2NBootstrapTripletDataset, mix_sources_from_args
from losses.charbonnier import CharbonnierLoss
from losses.rtv import RTVRegularizer
from models.denoiser import Denoiser
from utils.checkpoint import load_weights_flexible


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sample_shape(dataset, idx):
    if isinstance(dataset, Subset):
        return get_sample_shape(dataset.dataset, dataset.indices[idx])
    if hasattr(dataset, "get_sample_shape"):
        return dataset.get_sample_shape(idx)
    sample = dataset[idx]
    return tuple(sample[0].shape[-2:])


class ShapeBatchSampler:
    """复用原项目 crop_size=0 时的按尺寸组 batch 逻辑。"""

    def __init__(self, dataset, batch_size: int, shuffle: bool, seed: int = 42, max_pixels_per_batch: int = 0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_pixels_per_batch = int(max_pixels_per_batch)
        self.epoch = 0
        self.groups = defaultdict(list)
        self.shape_batch_sizes = {}

        print(f"[INFO] Building shape-grouped batches for {len(dataset)} samples...", flush=True)
        for idx in range(len(dataset)):
            self.groups[get_sample_shape(dataset, idx)].append(idx)

        for shape in self.groups:
            height, width = shape
            if self.max_pixels_per_batch > 0:
                shape_pixels = max(1, int(height) * int(width))
                shape_batch_size = max(1, self.max_pixels_per_batch // shape_pixels)
                self.shape_batch_sizes[shape] = min(self.batch_size, shape_batch_size)
            else:
                self.shape_batch_sizes[shape] = self.batch_size

        self.batch_count = sum(
            math.ceil(len(indices) / self.shape_batch_sizes[shape])
            for shape, indices in self.groups.items()
        )
        print(f"[INFO] Shape grouping ready: shapes={len(self.groups)}, batches={self.batch_count}", flush=True)

    def __len__(self):
        return self.batch_count

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for shape, indices in self.groups.items():
            indices = list(indices)
            shape_batch_size = self.shape_batch_sizes[shape]
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]
            for start in range(0, len(indices), shape_batch_size):
                batches.append(indices[start:start + shape_batch_size])
        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        self.epoch += 1
        return iter(batches)


class _PairView(torch.utils.data.Dataset):
    """把 N2NBootstrapTripletDataset 的三元组样本，降维成 N2N 需要的 (input, target) 噪声对。"""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def get_sample_shape(self, idx):
        return (self.base.crop_size, self.base.crop_size)

    def __getitem__(self, idx):
        d = self.base[idx]
        return d["input"], d["target"]


def build_multisource_loaders(args):
    """多被试（5x5 + mix 脑/腿）统一加载：与 D'/T 完全相同的数据源，保证公平对照。"""
    crop = args.crop_size if args.crop_size > 0 else 512
    base = N2NBootstrapTripletDataset(
        root_dir=args.data_path,
        intervals=args.intervals,
        crop_size=crop,
        random_crop=True,
        data_subdirs=("npy", "lbf"),
        strict_data_subdir=bool(args.strict_data_subdir),
        include_levels=tuple(args.levels) if args.levels else None,
        extra_sources=mix_sources_from_args(args),
        intensity_transform=args.intensity_transform,
        compute_pseudo_clean=False,  # N2N 只需噪声对，不需要伪干净
        augment=True,
    )
    view = _PairView(base)
    total = len(view)
    train_size = min(max(int(args.train_fraction * total), 1), total)
    val_size = total - train_size
    if val_size > 0:
        train_ds, val_ds = random_split(view, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds, val_ds = view, None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              prefetch_factor=2 if args.num_workers > 0 else None)
    val_loader = None if val_ds is None else DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_num_workers, pin_memory=True)
    print(f"[INFO] multi-source N2N: {total} samples (crop={crop}, 5x5 levels={args.levels}, "
          f"mix_root={args.mix_root}, mix_scenes={args.mix_scenes})")
    return base, train_loader, val_loader


def build_loaders(args):
    if args.mix_root:
        return build_multisource_loaders(args)
    lambda_conditioned = bool(args.lambda_conditioned) and args.intensity_transform == "boxcox"
    full_dataset = SpeckleN2NLogDataset(
        root_dir=args.data_path,
        crop_size=args.crop_size,
        intervals=args.intervals,
        boxcox_lam=args.boxcox_lam,
        boxcox_eps=args.boxcox_eps,
        lambda_conditioned=lambda_conditioned,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_candidates=args.lambda_candidates,
        intensity_transform=args.intensity_transform,
        vst_lut=args.vst_lut,
        npy_folder_name=args.data_subdir,
        strict_data_subdir=bool(args.strict_data_subdir),
        data_index_min=args.data_index_min,
        data_index_max=args.data_index_max,
        include_levels=tuple(args.levels) if args.levels else None,
    )

    total_size = len(full_dataset)
    train_size = int(args.train_fraction * total_size)
    train_size = min(max(train_size, 1), total_size)
    val_size = total_size - train_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset, val_dataset = full_dataset, None

    if args.crop_size <= 0:
        max_pixels_per_batch = args.max_pixels_per_batch or args.batch_size * args.batch_ref_size * args.batch_ref_size
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=ShapeBatchSampler(train_dataset, args.batch_size, shuffle=True, seed=args.seed, max_pixels_per_batch=max_pixels_per_batch),
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        val_loader = None if val_dataset is None else DataLoader(
            val_dataset,
            batch_sampler=ShapeBatchSampler(val_dataset, args.batch_size, shuffle=False, seed=args.seed + 1, max_pixels_per_batch=max_pixels_per_batch),
            num_workers=args.val_num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        val_loader = None if val_dataset is None else DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.val_num_workers,
            pin_memory=True,
        )
    return full_dataset, train_loader, val_loader


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
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    full_dataset, train_loader, val_loader = build_loaders(args)
    input_channels = 2 if bool(args.lambda_conditioned) and args.intensity_transform == "boxcox" else 1
    model = Denoiser(input_channels=input_channels).to(device)
    if args.init_checkpoint:
        info = load_weights_flexible(model, args.init_checkpoint, device)
        print(f"[INFO] Loaded init checkpoint: {info}")

    if torch.cuda.device_count() > 1 and bool(args.data_parallel):
        model = torch.nn.DataParallel(model)

    criterion_char = CharbonnierLoss().to(device)
    criterion_rtv = RTVRegularizer(radius=2, sigma=2.0, eps=1e-3).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_final,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = build_onecycle(optimizer, len(train_loader), args)
    writer = SummaryWriter(log_dir=str(log_dir))

    run_config = vars(args).copy()
    run_config.update({"dataset_size": len(full_dataset), "train_batches": len(train_loader)})
    (save_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO] device={device}, input_channels={input_channels}, dataset={len(full_dataset)}")
    print(f"[INFO] optimizer=AdamW, lr_final={args.lr_final}, lr_max={args.lr_max}, warmup_pct={args.warmup_pct}, scheduler=OneCycleLR(cos)")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        pbar = tqdm(train_loader, desc=f"N2N epoch {epoch}/{args.epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss_content = criterion_char(outputs, targets)
            loss_rtv = criterion_rtv(outputs)
            loss = loss_content + args.rtv_weight * loss_rtv

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            lr = scheduler.get_last_lr()[0]
            train_total += float(loss.item())
            writer.add_scalar("Train/Loss_Step", float(loss.item()), global_step)
            writer.add_scalar("Train/Learning_Rate", lr, global_step)
            global_step += 1
            pbar.set_postfix({"loss": f"{float(loss.item()):.6f}", "lr": f"{lr:.6g}"})

        avg_train = train_total / max(1, len(train_loader))
        writer.add_scalar("Train/Epoch_Avg_Loss", avg_train, epoch)

        avg_val = 0.0
        actual_val_steps = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for i, (inputs, targets) in tqdm(enumerate(val_loader), total=min(args.val_limit_batches, len(val_loader)), desc="Validating"):
                    if i >= args.val_limit_batches:
                        break
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss_content = criterion_char(outputs, targets)
                    loss_rtv = criterion_rtv(outputs)
                    loss = loss_content + args.rtv_weight * loss_rtv
                    avg_val += float(loss.item())
                    actual_val_steps += 1
                    writer.add_scalar("Val/Loss_Step", float(loss.item()), global_step)
                    global_step += 1
            avg_val = avg_val / actual_val_steps if actual_val_steps > 0 else 0.0
            writer.add_scalar("Val/Epoch_Avg_Loss", avg_val, epoch)

        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        save_path = save_dir / f"model_epoch_{epoch}.pth"
        torch.save(state_dict, save_path)
        print(f"[EPOCH {epoch}] train_loss={avg_train:.6f} val_loss={avg_val:.6f} saved={save_path}")

    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train N2N baseline with the original project schedule.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_subdir", type=str, default="npy")
    parser.add_argument("--strict_data_subdir", type=int, default=1)
    parser.add_argument("--data_index_min", type=int, default=-1)
    parser.add_argument("--data_index_max", type=int, default=-1)
    parser.add_argument("--levels", type=int, nargs="*", default=None,
                        help="只用这些叠加层级 5x5xN 的 N 训练（如 --levels 2 3 4 把 level1 留作 OOD 对照）。")
    parser.add_argument("--mix_root", type=str, default="",
                        help="额外数据根目录（如 /mnt2/songyd/mix）；给定则启用多被试统一加载，与 D'/T 同源。")
    parser.add_argument("--mix_scenes", type=str, nargs="*", default=None,
                        help="只取 mix_root 下这些场景编号（脑 305..312 + 腿 316..321；手 325 留作 OOD）。")
    parser.add_argument("--mix_subdirs", type=str, nargs="*", default=None,
                        help="mix 数据子目录（默认 npy+lbf 自动兼容）。")
    parser.add_argument("--intervals", type=int, nargs="*", default=[5, 7, 9])
    parser.add_argument("--save_dir", type=str, default="results/checkpoints/n2n")
    parser.add_argument("--log_dir", type=str, default="results/logs/n2n")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--max_pixels_per_batch", type=int, default=0)
    parser.add_argument("--batch_ref_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_fraction", type=float, default=0.99)
    parser.add_argument("--val_limit_batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01, help="Compatibility alias for lr_max.")
    parser.add_argument("--lr_max", type=float, default=None)
    parser.add_argument("--lr_final", type=float, default=0.0005)
    parser.add_argument("--warmup_pct", type=float, default=0.1)
    parser.add_argument("--intensity_transform", choices=["log1p", "boxcox", "learned_vst"], default="log1p")
    parser.add_argument("--vst_lut", type=str, default="")
    parser.add_argument("--boxcox_lam", type=float, default=-0.15)
    parser.add_argument("--boxcox_eps", type=float, default=1e-6)
    parser.add_argument("--lambda_conditioned", type=int, default=0)
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--lambda_min", type=float, default=-0.3)
    parser.add_argument("--lambda_max", type=float, default=0.2)
    parser.add_argument(
        "--lambda_candidates",
        type=float,
        nargs="*",
        default=[-0.3, -0.25, -0.2, -0.15, -0.1, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    )
    parser.add_argument("--rtv_weight", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--data_parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    if args.data_index_min < 0:
        args.data_index_min = None
    if args.data_index_max < 0:
        args.data_index_max = None
    if args.lr_max is None:
        args.lr_max = args.lr
    if args.lr_max <= args.lr_final:
        raise ValueError(f"lr_max must be greater than lr_final, got lr_max={args.lr_max}, lr_final={args.lr_final}")
    return args


if __name__ == "__main__":
    train(parse_args())
