# -*- coding: utf-8 -*-
"""Train Masked N2N + masked feature prediction on aligned noisy BFI pairs.

Normal branch:
    n1 -> student -> y1,                    L_n2n = Charb(y1, n2)
Masked branch:
    mask(n1) -> student -> y_mask/F_mask,   L_mask_pixel on hidden pixels
Teacher branch (only when feature weight > 0):
    n2 -> EMA teacher -> F_target,           L_mask_feature on hidden locations

The old full-image F(n1)~=F(n2) consistency loss is intentionally not used.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from train_n2n import build_loaders, build_onecycle
from models.masked_denoiser import MaskedDenoiserWithFeats, FEAT_CHANNELS
from losses.charbonnier import CharbonnierLoss
from losses.rtv import RTVRegularizer
from losses.masked_prediction import (
    MaskedFeaturePredictionLoss,
    apply_visible_mask,
    make_block_visible_mask,
    masked_charbonnier,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


@contextmanager
def suspend_batchnorm_running_stats(module: nn.Module):
    """Masked forward 使用 batch statistics，但不写入推理期 running statistics。"""
    batchnorms = [
        child for child in unwrap(module).modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ]
    previous = [child.track_running_stats for child in batchnorms]
    try:
        for child in batchnorms:
            child.track_running_stats = False
        yield
    finally:
        for child, track_running_stats in zip(batchnorms, previous):
            child.track_running_stats = track_running_stats


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """EMA parameters; copy BN/statistic buffers from the latest student."""
    source = unwrap(student)
    target = unwrap(teacher)
    source_params = dict(source.named_parameters())
    for name, target_param in target.named_parameters():
        target_param.mul_(decay).add_(source_params[name].detach(), alpha=1.0 - decay)
    source_buffers = dict(source.named_buffers())
    for name, target_buffer in target.named_buffers():
        target_buffer.copy_(source_buffers[name])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Masked N2N + masked feature prediction")
    # Data arguments consumed by train_n2n.build_loaders.
    p.add_argument("--data_path", required=True)
    p.add_argument("--data_subdir", default="npy")
    p.add_argument("--strict_data_subdir", type=int, default=1)
    p.add_argument("--data_index_min", type=int, default=-1)
    p.add_argument("--data_index_max", type=int, default=-1)
    p.add_argument("--levels", type=int, nargs="*", default=None)
    p.add_argument("--mix_root", default="")
    p.add_argument("--mix_scenes", nargs="*", default=None)
    p.add_argument("--mix_subdirs", nargs="*", default=None)
    p.add_argument("--intervals", type=int, nargs="*", default=[5, 7, 9])
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_pixels_per_batch", type=int, default=0)
    p.add_argument("--batch_ref_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_num_workers", type=int, default=4)
    p.add_argument("--train_fraction", type=float, default=0.99)
    p.add_argument("--val_limit_batches", type=int, default=20)
    p.add_argument("--intensity_transform", choices=["log1p", "boxcox", "learned_vst"], default="log1p")
    p.add_argument("--vst_lut", default="")
    p.add_argument("--boxcox_lam", type=float, default=-0.15)
    p.add_argument("--boxcox_eps", type=float, default=1e-6)
    p.add_argument("--lambda_conditioned", type=int, default=0)
    p.add_argument("--lambda_min", type=float, default=-0.3)
    p.add_argument("--lambda_max", type=float, default=0.2)
    p.add_argument("--lambda_candidates", type=float, nargs="*", default=[-0.3, -0.2, -0.15, -0.1, 0.0, 0.1, 0.2])

    # Optimization.
    p.add_argument("--save_dir", default="results/checkpoints/masked_n2n")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01, help="alias for lr_max")
    p.add_argument("--lr_max", type=float, default=None)
    p.add_argument("--lr_final", type=float, default=0.0005)
    p.add_argument("--warmup_pct", type=float, default=0.1)
    p.add_argument("--rtv_weight", type=float, default=0.01)
    p.add_argument("--charb_eps", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--data_parallel", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")

    # Masked objectives.
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--mask_patch", type=int, default=16)
    p.add_argument("--mask_fill", choices=["zero", "mean"], default="zero")
    p.add_argument("--w_mask_pixel", type=float, default=0.1)
    p.add_argument("--w_mask_feature", type=float, default=0.05)
    p.add_argument("--mask_feature_scales", nargs="*", default=["encoder2", "encoder3"],
                   choices=["encoder1", "encoder2", "encoder3", "bottleneck"])
    p.add_argument("--mask_feature_weights", type=float, nargs="*", default=None)
    p.add_argument("--predictor_hidden_ratio", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.996)
    p.add_argument("--feature_warmup_frac", type=float, default=0.1)
    p.add_argument("--freeze_masked_bn_stats", type=int, default=1,
                   help="masked forward 不更新 student BatchNorm running statistics")
    p.add_argument("--deterministic_loader_rng", type=int, default=1,
                   help="DataLoader 使用独立 generator，避免模型随机数改变样本顺序/worker crop")
    p.add_argument("--mask_seed_offset", type=int, default=20_001,
                   help="mask generator seed = seed + offset")
    p.add_argument("--predictor_seed_offset", type=int, default=30_001,
                   help="feature predictor seed = seed + offset，且不推进全局 Torch RNG")

    args = p.parse_args()
    args.data_index_min = None if args.data_index_min < 0 else args.data_index_min
    args.data_index_max = None if args.data_index_max < 0 else args.data_index_max
    args.lr_max = args.lr if args.lr_max is None else args.lr_max
    if args.lr_max <= args.lr_final:
        raise ValueError("lr_max must be greater than lr_final")
    if not 0.0 <= args.mask_ratio < 1.0 or args.mask_patch <= 0:
        raise ValueError("mask_ratio must be in [0,1) and mask_patch must be positive")
    if args.w_mask_pixel < 0 or args.w_mask_feature < 0:
        raise ValueError("masked loss weights must be non-negative")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0,1)")
    if args.feature_warmup_frac < 0:
        raise ValueError("feature_warmup_frac must be non-negative")
    if args.mask_seed_offset < 0 or args.predictor_seed_offset < 0:
        raise ValueError("mask/predictor seed offsets must be non-negative")
    if args.w_mask_feature > 0 and not args.mask_feature_scales:
        raise ValueError("feature scales cannot be empty when masked feature loss is enabled")
    if args.lambda_conditioned:
        raise ValueError("train_masked.py currently supports one image channel plus one mask channel; "
                         "lambda-conditioned Box-Cox input is not supported")
    if args.mask_feature_weights is not None and len(args.mask_feature_weights) != len(args.mask_feature_scales):
        raise ValueError("mask_feature_weights must match mask_feature_scales")
    return args


@torch.no_grad()
def validate(model, loader, charb, rtv, args, device) -> float:
    if loader is None:
        return 0.0
    model.eval()
    total = 0.0
    steps = 0
    for i, (n1, n2) in enumerate(loader):
        if i >= args.val_limit_batches:
            break
        n1 = n1.to(device, non_blocking=True)
        n2 = n2.to(device, non_blocking=True)
        y = model(n1)  # omitted mask => all visible
        loss = charb(y, n2) + args.rtv_weight * rtv(y)
        total += float(loss)
        steps += 1
    return total / max(1, steps)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _, train_loader, val_loader = build_loaders(args)
    student: nn.Module = MaskedDenoiserWithFeats(image_channels=1).to(device)
    teacher: nn.Module | None = None
    if args.w_mask_feature > 0:
        teacher = copy.deepcopy(student).to(device).eval()
        teacher.requires_grad_(False)

    scale_info = {
        "encoder1": (0, FEAT_CHANNELS[0]),
        "encoder2": (1, FEAT_CHANNELS[1]),
        "encoder3": (2, FEAT_CHANNELS[2]),
        "bottleneck": (3, FEAT_CHANNELS[3]),
    }
    feat_idx = [scale_info[name][0] for name in args.mask_feature_scales]
    feat_channels = [scale_info[name][1] for name in args.mask_feature_scales]
    feature_loss = None
    if args.w_mask_feature > 0:
        # predictor 初始化使用独立 seed，并在退出后恢复全局 CPU RNG 状态；
        # 因此 C/D 不会仅因多建了 predictor 就改变后续数据随机轨迹。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + args.predictor_seed_offset)
            feature_loss = MaskedFeaturePredictionLoss(
                feat_channels,
                weights=args.mask_feature_weights,
                predictor_hidden_ratio=args.predictor_hidden_ratio,
            ).to(device)

    if args.data_parallel and torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)
        if teacher is not None:
            teacher = nn.DataParallel(teacher)

    charb = CharbonnierLoss(eps=args.charb_eps).to(device)
    rtv = RTVRegularizer(radius=2, sigma=2.0, eps=1e-3).to(device)
    trainable = list(student.parameters()) + (list(feature_loss.parameters()) if feature_loss is not None else [])
    optimizer = optim.AdamW(trainable, lr=args.lr_max, weight_decay=1e-4)
    scheduler = build_onecycle(optimizer, len(train_loader), args)

    total_steps = max(1, args.epochs * len(train_loader))
    feature_warmup_steps = int(max(0.0, args.feature_warmup_frac) * total_steps)
    use_mask = args.w_mask_pixel > 0 or args.w_mask_feature > 0
    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(args.seed + args.mask_seed_offset)
    print(
        f"[INFO] device={device} batches={len(train_loader)} mask={use_mask} "
        f"ratio={args.mask_ratio} patch={args.mask_patch} fill={args.mask_fill}"
    )
    print(
        f"[INFO] loss = N2N + {args.w_mask_pixel}*mask_pixel + "
        f"{args.w_mask_feature}*mask_feature + {args.rtv_weight}*RTV; "
        f"feature_scales={args.mask_feature_scales} ema={args.ema_decay}"
    )
    print(
        f"[INFO] controls: freeze_masked_bn_stats={bool(args.freeze_masked_bn_stats)} "
        f"deterministic_loader_rng={bool(args.deterministic_loader_rng)} "
        f"mask_seed={args.seed + args.mask_seed_offset} "
        f"predictor_seed={args.seed + args.predictor_seed_offset}"
    )

    history_path = save_dir / "history.jsonl"
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        if feature_loss is not None:
            feature_loss.train()
        running: dict[str, float] = {}
        pbar = tqdm(train_loader, desc=f"Masked N2N {epoch}/{args.epochs}")
        for n1, n2 in pbar:
            n1 = n1.to(device, non_blocking=True)
            n2 = n2.to(device, non_blocking=True)
            loss_mask_pixel = n1.new_zeros(())
            loss_mask_feature = n1.new_zeros(())
            ramp = 0.0
            per_scale: list[float] = []
            actual_hidden = 0.0
            if use_mask:
                visible = make_block_visible_mask(
                    n1.shape[0], n1.shape[2], n1.shape[3], args.mask_ratio, args.mask_patch,
                    device=n1.device, dtype=n1.dtype, generator=mask_generator,
                )
                actual_hidden = float((1.0 - visible).mean())
                masked_n1 = apply_visible_mask(n1, visible, fill=args.mask_fill)
                # Masked 分支仍使用当前 batch statistics 和 BN affine 参数梯度，
                # 但不允许它污染 all-visible 推理所依赖的 running statistics。
                bn_context = (
                    suspend_batchnorm_running_stats(student)
                    if args.freeze_masked_bn_stats else nullcontext()
                )
                with bn_context:
                    y_masked, masked_feats = student(masked_n1, visible, return_feats=True)
                if args.w_mask_pixel > 0:
                    loss_mask_pixel = masked_charbonnier(
                        y_masked, n2, visible, eps=args.charb_eps
                    )
                if feature_loss is not None and teacher is not None:
                    with torch.no_grad():
                        _, target_feats = teacher(n2, return_feats=True)
                    student_selected = [masked_feats[i] for i in feat_idx]
                    target_selected = [target_feats[i] for i in feat_idx]
                    loss_mask_feature, per_scale = feature_loss(
                        student_selected, target_selected, visible
                    )
                    ramp = 1.0 if feature_warmup_steps <= 0 else min(
                        1.0, float(global_step + 1) / feature_warmup_steps
                    )

            # 只有 all-visible 分支更新持久化 BN statistics。
            y_normal = student(n1)  # all-visible mask is injected by the wrapper
            loss_n2n = charb(y_normal, n2)
            loss_rtv = rtv(y_normal) if args.rtv_weight > 0 else y_normal.new_zeros(())
            weighted_rtv = args.rtv_weight * loss_rtv
            weighted_mask_pixel = args.w_mask_pixel * loss_mask_pixel
            weighted_mask_feature = args.w_mask_feature * ramp * loss_mask_feature
            total = loss_n2n + weighted_rtv + weighted_mask_pixel + weighted_mask_feature

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()
            if teacher is not None:
                update_ema(student, teacher, args.ema_decay)
                teacher.eval()
            global_step += 1

            logs = {
                "total": float(total.detach()),
                "n2n": float(loss_n2n.detach()),
                "rtv": float(loss_rtv.detach()),
                "mask_pixel": float(loss_mask_pixel.detach()),
                "mask_feature": float(loss_mask_feature.detach()),
                "weighted_rtv": float(weighted_rtv.detach()),
                "weighted_mask_pixel": float(weighted_mask_pixel.detach()),
                "weighted_mask_feature": float(weighted_mask_feature.detach()),
                "hidden": actual_hidden,
            }
            for i, value in enumerate(per_scale):
                logs[f"mask_feat_{args.mask_feature_scales[i]}"] = value
            for key, value in logs.items():
                running[key] = running.get(key, 0.0) + value
            pbar.set_postfix({
                "loss": f"{logs['total']:.4f}",
                "wpix": f"{logs['weighted_mask_pixel']:.4f}",
                "wfeat": f"{logs['weighted_mask_feature']:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2g}",
            })

        avg = {key: value / max(1, len(train_loader)) for key, value in running.items()}
        val = validate(student, val_loader, charb, rtv, args, device)
        record = {"epoch": epoch, "val": val, **avg}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        payload = {
            "model": unwrap(student).state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }
        if feature_loss is not None:
            payload["feature_predictor"] = feature_loss.state_dict()
        if teacher is not None:
            payload["teacher"] = unwrap(teacher).state_dict()
        save_path = save_dir / f"model_epoch_{epoch}.pth"
        torch.save(payload, save_path)
        fields = " ".join(f"{k}={v:.5f}" for k, v in record.items() if k != "epoch")
        print(f"[EPOCH {epoch}] {fields} saved={save_path}")


if __name__ == "__main__":
    train(parse_args())
