# -*- coding: utf-8 -*-
"""Train N2N with region-based masked or Gaussian feature prediction.

Normal branch:
    n1 -> student -> y1,                    L_n2n = Charb(y1, n2)
Masked branch:
    mask(n1) -> student -> y_mask/F_mask,   L_mask_pixel on hidden pixels
Gaussian branch:
    n1 + local noise -> single-channel student -> F_noise,
    L_mask_feature on perturbed locations (the region map is not a model input)
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
from models.denoiser_feats import DenoiserWithFeats, FEAT_CHANNELS
from models.masked_denoiser import MaskedDenoiserWithFeats
from losses.charbonnier import CharbonnierLoss
from losses.rtv import RTVRegularizer
from losses.masked_prediction import (
    MaskedFeaturePredictionLoss,
    apply_local_gaussian_noise,
    apply_visible_mask,
    make_block_visible_mask,
    masked_charbonnier,
)
from utils.training_curves import update_training_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


@contextmanager
def suspend_batchnorm_running_stats(module: nn.Module):
    """辅助扰动 forward 使用 batch statistics，但不写入推理期 running statistics。"""
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


def compute_gradient_diagnostics(
    student: nn.Module,
    loss_n2n: torch.Tensor,
    weighted_feature_loss: torch.Tensor,
    scales: list[str],
) -> dict[str, dict[str, float]]:
    """只读计算两项目标在指定编码层的梯度强度和夹角。

    使用 ``torch.autograd.grad`` 返回临时梯度，不写入参数的 ``.grad``，因此随后执行的
    ``total.backward()`` 与不开诊断时完全相同。诊断只增加少量计算和显存开销。
    """

    backbone = unwrap(student)
    modules = {
        "encoder1": backbone.encoder.Light_Residual_block_1,
        "encoder2": backbone.encoder.Light_Residual_block_2,
        "encoder3": backbone.encoder.Light_Residual_block_3,
        "bottleneck": backbone.bridge,
    }
    unknown = [name for name in scales if name not in modules]
    if unknown:
        raise ValueError(f"unknown gradient diagnostic scales: {unknown}")

    output: dict[str, dict[str, float]] = {}
    for name in scales:
        parameters = [parameter for parameter in modules[name].parameters() if parameter.requires_grad]
        grads_n2n = torch.autograd.grad(
            loss_n2n, parameters, retain_graph=True, allow_unused=True
        )
        grads_feature = torch.autograd.grad(
            weighted_feature_loss, parameters, retain_graph=True, allow_unused=True
        )

        # 用 float64 累加诊断标量，避免半精度或大特征图导致统计精度不足。
        device = loss_n2n.device
        n2n_sq = torch.zeros((), device=device, dtype=torch.float64)
        feature_sq = torch.zeros((), device=device, dtype=torch.float64)
        dot = torch.zeros((), device=device, dtype=torch.float64)
        for grad_n2n, grad_feature in zip(grads_n2n, grads_feature):
            if grad_n2n is not None:
                grad_n2n = grad_n2n.detach().to(dtype=torch.float64)
                n2n_sq = n2n_sq + grad_n2n.square().sum()
            if grad_feature is not None:
                grad_feature = grad_feature.detach().to(dtype=torch.float64)
                feature_sq = feature_sq + grad_feature.square().sum()
            if grad_n2n is not None and grad_feature is not None:
                dot = dot + (grad_n2n * grad_feature).sum()

        norm_n2n = n2n_sq.sqrt()
        norm_feature = feature_sq.sqrt()
        denominator = (norm_n2n * norm_feature).clamp_min(1e-30)
        output[name] = {
            "n2n_norm": float(norm_n2n.cpu()),
            "weighted_feature_norm": float(norm_feature.cpu()),
            "feature_to_n2n_ratio": float((norm_feature / norm_n2n.clamp_min(1e-30)).cpu()),
            "cosine": float((dot / denominator).clamp(-1.0, 1.0).cpu()),
        }
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="N2N + region-based masked/Gaussian feature prediction")
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
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay；默认保持 masked A/C 既有配方")
    p.add_argument("--charb_eps", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--data_parallel", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")

    # Region-based auxiliary objectives. The default keeps historical mask runs unchanged.
    p.add_argument("--corruption_mode", choices=["mask", "gaussian"], default="mask",
                   help="mask=历史双通道硬遮挡；gaussian=单通道局部加噪，区域图不输入网络")
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
    p.add_argument("--noise_seed_offset", type=int, default=40_001,
                   help="Gaussian noise generator seed = seed + offset")
    p.add_argument("--noise_stats_json", default="",
                   help="measure_noise.py 输出；gaussian 模式默认读取 recommended_sigma")
    p.add_argument("--noise_sigma", type=float, default=0.0,
                   help="log1p 域参考 sigma；>0 时覆盖 --noise_stats_json")
    p.add_argument("--noise_sigma_min_scale", type=float, default=0.25,
                   help="每张图 sigma 下界 = reference sigma * 此比例")
    p.add_argument("--noise_sigma_max_scale", type=float, default=0.75,
                   help="每张图 sigma 上界 = reference sigma * 此比例")
    p.add_argument("--predictor_seed_offset", type=int, default=30_001,
                   help="feature predictor seed = seed + offset，且不推进全局 Torch RNG")
    p.add_argument("--grad_diag_every", type=int, default=0,
                   help="每隔多少 optimizer step 记录一次分目标梯度；0=关闭（推荐微调时设 100）")
    p.add_argument("--grad_diag_scales", nargs="*", default=["encoder2", "encoder3"],
                   choices=["encoder1", "encoder2", "encoder3", "bottleneck"],
                   help="梯度诊断作用层；不改变实际 loss 或反向传播")
    p.add_argument("--plot_loss_curve", type=int, default=1,
                   help="1=每个 epoch 自动更新 loss_curve.png 和 loss_history.csv")

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
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0,1)")
    if args.feature_warmup_frac < 0:
        raise ValueError("feature_warmup_frac must be non-negative")
    if args.mask_seed_offset < 0 or args.noise_seed_offset < 0 or args.predictor_seed_offset < 0:
        raise ValueError("mask/noise/predictor seed offsets must be non-negative")
    if args.noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")
    if (args.noise_sigma_min_scale < 0
            or args.noise_sigma_max_scale < args.noise_sigma_min_scale):
        raise ValueError("noise sigma scales must satisfy 0 <= min <= max")
    if args.corruption_mode == "gaussian" and args.w_mask_pixel > 0:
        raise ValueError("gaussian mode does not use pixel inpainting; set --w_mask_pixel 0")
    if args.corruption_mode == "gaussian" and args.w_mask_feature <= 0:
        raise ValueError("gaussian mode requires --w_mask_feature > 0")
    if (args.corruption_mode == "gaussian" and args.noise_sigma <= 0
            and not args.noise_stats_json):
        raise ValueError("gaussian mode requires --noise_sigma > 0 or --noise_stats_json")
    if args.grad_diag_every < 0:
        raise ValueError("grad_diag_every must be non-negative")
    if args.grad_diag_every > 0 and not args.grad_diag_scales:
        raise ValueError("grad_diag_scales cannot be empty when diagnostics are enabled")
    if args.w_mask_feature > 0 and not args.mask_feature_scales:
        raise ValueError("feature scales cannot be empty when masked feature loss is enabled")
    if args.lambda_conditioned:
        raise ValueError("train_masked.py does not support lambda-conditioned Box-Cox input")
    if args.mask_feature_weights is not None and len(args.mask_feature_weights) != len(args.mask_feature_scales):
        raise ValueError("mask_feature_weights must match mask_feature_scales")
    return args


def resolve_noise_sigma(args: argparse.Namespace) -> tuple[float, float, float]:
    """解析 Gaussian 模式在模型输入域使用的参考/最小/最大 sigma。"""
    if args.corruption_mode != "gaussian":
        return 0.0, 0.0, 0.0

    reference = float(args.noise_sigma)
    if reference <= 0.0:
        stats_path = Path(args.noise_stats_json)
        if not stats_path.is_file():
            raise FileNotFoundError(f"noise statistics not found: {stats_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats_transform = stats.get("intensity_transform")
        if stats_transform and stats_transform != args.intensity_transform:
            raise ValueError(
                f"noise stats domain {stats_transform!r} != training domain "
                f"{args.intensity_transform!r}"
            )
        reference = float(stats["recommended_sigma"])
    if reference <= 0.0:
        raise ValueError(f"resolved noise sigma must be positive, got {reference}")
    return (
        reference,
        reference * float(args.noise_sigma_min_scale),
        reference * float(args.noise_sigma_max_scale),
    )


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
        y = model(n1)  # mask 模式自动补全 all-visible；Gaussian 模式直接使用单通道输入。
        loss = charb(y, n2) + args.rtv_weight * rtv(y)
        total += float(loss)
        steps += 1
    return total / max(1, steps)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    noise_sigma_ref, noise_sigma_min, noise_sigma_max = resolve_noise_sigma(args)
    args.resolved_noise_sigma = noise_sigma_ref
    args.resolved_noise_sigma_min = noise_sigma_min
    args.resolved_noise_sigma_max = noise_sigma_max
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _, train_loader, val_loader = build_loaders(args)
    # Gaussian 模式恢复真正的单通道结构；随机区域只供扰动和 loss 使用，不喂给模型。
    student: nn.Module = (
        DenoiserWithFeats(input_channels=1)
        if args.corruption_mode == "gaussian"
        else MaskedDenoiserWithFeats(image_channels=1)
    ).to(device)
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
    optimizer = optim.AdamW(trainable, lr=args.lr_max, weight_decay=args.weight_decay)
    scheduler = build_onecycle(optimizer, len(train_loader), args)

    total_steps = max(1, args.epochs * len(train_loader))
    feature_warmup_steps = int(max(0.0, args.feature_warmup_frac) * total_steps)
    use_corruption = args.w_mask_pixel > 0 or args.w_mask_feature > 0
    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(args.seed + args.mask_seed_offset)
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(args.seed + args.noise_seed_offset)
    print(
        f"[INFO] device={device} batches={len(train_loader)} corruption={args.corruption_mode} "
        f"enabled={use_corruption} ratio={args.mask_ratio} patch={args.mask_patch}"
    )
    if args.corruption_mode == "mask":
        print(f"[INFO] mask fill={args.mask_fill}; model input=image+visibility channel")
    else:
        print(
            f"[INFO] local Gaussian sigma(log-domain) reference={noise_sigma_ref:.6g} "
            f"range=[{noise_sigma_min:.6g}, {noise_sigma_max:.6g}]; "
            "model input=image only"
        )
    print(
        f"[INFO] loss = N2N + {args.w_mask_pixel}*mask_pixel + "
        f"{args.w_mask_feature}*mask_feature + {args.rtv_weight}*RTV; "
        f"feature_scales={args.mask_feature_scales} ema={args.ema_decay}"
    )
    print(
        f"[INFO] controls: freeze_masked_bn_stats={bool(args.freeze_masked_bn_stats)} "
        f"deterministic_loader_rng={bool(args.deterministic_loader_rng)} "
        f"weight_decay={args.weight_decay} "
        f"mask_seed={args.seed + args.mask_seed_offset} "
        f"noise_seed={args.seed + args.noise_seed_offset} "
        f"predictor_seed={args.seed + args.predictor_seed_offset} "
        f"grad_diag_every={args.grad_diag_every} grad_diag_scales={args.grad_diag_scales}"
    )

    history_path = save_dir / "history.jsonl"
    grad_diagnostics_path = save_dir / "grad_diagnostics.jsonl"
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        if feature_loss is not None:
            feature_loss.train()
        running: dict[str, float] = {}
        pbar = tqdm(train_loader, desc=f"{args.corruption_mode.title()} feature N2N {epoch}/{args.epochs}")
        for batch_index, (n1, n2) in enumerate(pbar):
            n1 = n1.to(device, non_blocking=True)
            n2 = n2.to(device, non_blocking=True)
            loss_mask_pixel = n1.new_zeros(())
            loss_mask_feature = n1.new_zeros(())
            ramp = 0.0
            per_scale: list[float] = []
            actual_hidden = 0.0
            sampled_noise_sigma = 0.0
            if use_corruption:
                visible = make_block_visible_mask(
                    n1.shape[0], n1.shape[2], n1.shape[3], args.mask_ratio, args.mask_patch,
                    device=n1.device, dtype=n1.dtype, generator=mask_generator,
                )
                actual_hidden = float((1.0 - visible).mean())
                if args.corruption_mode == "mask":
                    corrupted_n1 = apply_visible_mask(n1, visible, fill=args.mask_fill)
                else:
                    corrupted_n1, sampled_sigmas = apply_local_gaussian_noise(
                        n1,
                        visible,
                        noise_sigma_min,
                        noise_sigma_max,
                        generator=noise_generator,
                        clamp_min=0.0,
                    )
                    sampled_noise_sigma = float(sampled_sigmas.mean())

                # 辅助分支仍使用当前 batch statistics 和 BN affine 参数梯度，
                # 但不允许它污染 all-visible 推理所依赖的 running statistics。
                bn_context = (
                    suspend_batchnorm_running_stats(student)
                    if args.freeze_masked_bn_stats else nullcontext()
                )
                with bn_context:
                    if args.corruption_mode == "mask":
                        y_masked, masked_feats = student(
                            corrupted_n1, visible, return_feats=True
                        )
                    else:
                        y_masked, masked_feats = student(corrupted_n1, return_feats=True)
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

            # 只有正常 N2N 分支更新持久化 BN statistics。
            y_normal = student(n1)  # mask 模式 wrapper 会自动补全 all-visible channel。
            loss_n2n = charb(y_normal, n2)
            loss_rtv = rtv(y_normal) if args.rtv_weight > 0 else y_normal.new_zeros(())
            weighted_rtv = args.rtv_weight * loss_rtv
            weighted_mask_pixel = args.w_mask_pixel * loss_mask_pixel
            weighted_mask_feature = args.w_mask_feature * ramp * loss_mask_feature
            total = loss_n2n + weighted_rtv + weighted_mask_pixel + weighted_mask_feature

            # 分目标梯度只做低频只读统计：autograd.grad 不写入 parameter.grad，
            # 因而不会改变下面 total.backward() 的优化结果。
            if (
                args.grad_diag_every > 0
                and feature_loss is not None
                and weighted_mask_feature.requires_grad
                and global_step % args.grad_diag_every == 0
            ):
                grad_record = {
                    "epoch": epoch,
                    "batch": batch_index,
                    "global_step": global_step,
                    "ramp": ramp,
                    "w_mask_feature": args.w_mask_feature,
                    "loss_n2n": float(loss_n2n.detach()),
                    "weighted_mask_feature": float(weighted_mask_feature.detach()),
                    "scales": compute_gradient_diagnostics(
                        student,
                        loss_n2n,
                        weighted_mask_feature,
                        args.grad_diag_scales,
                    ),
                }
                with grad_diagnostics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(grad_record, ensure_ascii=False) + "\n")

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
                "noise_sigma": sampled_noise_sigma,
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
        if args.plot_loss_curve:
            try:
                curve_title = (
                    "Local-Gaussian N2N with feature prediction"
                    if args.corruption_mode == "gaussian"
                    else "Masked N2N with feature prediction"
                )
                update_training_curves(history_path, curve_title)
            except Exception as error:
                # 绘图是只读诊断，不能因为可视化异常中断长时间训练。
                print(f"[WARN] loss 曲线更新失败，训练继续：{error}")

        payload = {
            "model": unwrap(student).state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "model_type": (
                "noise_feature_single_channel"
                if args.corruption_mode == "gaussian"
                else "masked_feature_two_channel"
            ),
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
