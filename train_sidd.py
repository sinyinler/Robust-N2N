"""SIDD-Small sRGB 的 scene-disjoint 监督训练入口。"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.sidd_dataset import SIDDSceneCropDataset, seed_sidd_worker
from losses.charbonnier import CharbonnierLoss
from losses.masked_prediction import (
    MaskedFeaturePredictionLoss,
    apply_local_gaussian_noise,
    make_block_visible_mask,
)
from losses.rtv import RTVRegularizer
from models.sidd_rgb_denoiser import (
    SIDD_FEAT_CHANNELS,
    SIDD_FEAT_NAMES,
    SIDDRGBDenoiser,
)
from utils.training_curves import update_training_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cosine_with_warmup(total_steps: int, warmup_steps: int, final_ratio: float):
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def flatten_crops(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    noisy = batch["noisy"].flatten(0, 1).to(device, non_blocking=True)
    gt = batch["gt"].flatten(0, 1).to(device, non_blocking=True)
    return noisy, gt


@contextmanager
def suspend_batchnorm_running_stats(module: torch.nn.Module):
    """辅助加噪分支使用 batch statistics，但不污染正常推理的 running statistics。"""

    batch_norms = [layer for layer in module.modules() if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [layer.momentum for layer in batch_norms]
    tracked = [layer.track_running_stats for layer in batch_norms]
    try:
        for layer in batch_norms:
            layer.track_running_stats = False
        yield
    finally:
        for layer, momentum, track in zip(batch_norms, momenta, tracked):
            layer.momentum = momentum
            layer.track_running_stats = track


@torch.no_grad()
def update_ema(student: torch.nn.Module, teacher: torch.nn.Module, decay: float) -> None:
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.mul_(decay).add_(student_param, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


@torch.inference_mode()
def validate(model, loader, criterion, device, amp: bool) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        noisy, gt = flatten_crops(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            loss = criterion(model(noisy), gt)
        total += float(loss.item()) * noisy.shape[0]
        count += noisy.shape[0]
    return total / max(count, 1)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    config: dict,
    val_loss: float,
    feature_loss=None,
    teacher=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "val_loss": val_loss,
    }
    if feature_loss is not None:
        payload["feature_predictor"] = feature_loss.state_dict()
    if teacher is not None:
        payload["teacher"] = teacher.state_dict()
    torch.save(payload, path)


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.data_root:
        config["data_root"] = args.data_root
    if args.out_dir:
        config["out_dir"] = args.out_dir
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.smoke_steps is not None:
        config["smoke_steps"] = args.smoke_steps

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    out_dir = Path(config["out_dir"])
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"输出目录非空，拒绝混写实验轨迹: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_dataset = SIDDSceneCropDataset(
        config["data_root"],
        config["train_scenes"],
        crop_size=config["crop_size"],
        repeats_per_pair=config["train_repeats_per_pair"],
        crops_per_load=config["crops_per_load"],
        augment=True,
        deterministic=False,
        seed=seed,
    )
    val_dataset = SIDDSceneCropDataset(
        config["data_root"],
        config["val_scenes"],
        crop_size=config["crop_size"],
        repeats_per_pair=config["val_repeats_per_pair"],
        crops_per_load=config["crops_per_load"],
        augment=False,
        deterministic=True,
        seed=seed + 100_000,
    )
    generator = torch.Generator().manual_seed(seed + 10_001)
    loader_kwargs = {
        "batch_size": int(config["image_batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_sidd_worker,
        "persistent_workers": int(config["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = SIDDRGBDenoiser().to(device)
    criterion = CharbonnierLoss(eps=float(config.get("charbonnier_eps", 1e-3))).to(device)
    feature_weight = float(config.get("feature_weight", 0.0))
    rtv_weight = float(config.get("rtv_weight", 0.0))
    if feature_weight < 0 or rtv_weight < 0:
        raise ValueError("feature_weight 和 rtv_weight 必须非负")

    feature_loss = None
    teacher = None
    feature_indices: list[int] = []
    if feature_weight > 0:
        selected_names = list(config.get("feature_scales", ["encoder2", "encoder3"]))
        unknown = sorted(set(selected_names) - set(SIDD_FEAT_NAMES))
        if unknown:
            raise ValueError(f"未知 feature scales: {unknown}; 可选 {SIDD_FEAT_NAMES}")
        feature_indices = [SIDD_FEAT_NAMES.index(name) for name in selected_names]
        channels = [SIDD_FEAT_CHANNELS[index] for index in feature_indices]
        feature_loss = MaskedFeaturePredictionLoss(
            channels,
            weights=config.get("feature_scale_weights"),
            predictor_hidden_ratio=float(config.get("predictor_hidden_ratio", 1.0)),
        ).to(device)
        teacher = copy.deepcopy(model).to(device).eval().requires_grad_(False)

    rtv = RTVRegularizer(
        radius=int(config.get("rtv_radius", 2)),
        sigma=float(config.get("rtv_sigma", 2.0)),
        eps=float(config.get("rtv_eps", 1e-3)),
    ).to(device)
    trainable = list(model.parameters())
    if feature_loss is not None:
        trainable += list(feature_loss.parameters())
    optimizer = AdamW(
        trainable, lr=float(config["lr_max"]), weight_decay=float(config["weight_decay"])
    )
    epochs = int(config["epochs"])
    smoke_steps = int(config.get("smoke_steps", 0))
    steps_per_epoch = min(len(train_loader), smoke_steps) if smoke_steps > 0 else len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = LambdaLR(
        optimizer,
        cosine_with_warmup(
            total_steps,
            round(total_steps * float(config.get("warmup_fraction", 0.05))),
            float(config["lr_final"]) / float(config["lr_max"]),
        ),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    feature_warmup_steps = round(total_steps * float(config.get("feature_warmup_fraction", 0.1)))
    mask_generator = torch.Generator(device=device).manual_seed(seed + 20_003)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 30_007)

    print(
        f"[SIDD] device={device}, train_pairs={len(train_dataset.pairs)}, "
        f"val_pairs={len(val_dataset.pairs)}, effective_patch_batch="
        f"{config['image_batch_size'] * config['crops_per_load']}, params="
        f"{sum(p.numel() for p in model.parameters()):,}, feature_weight={feature_weight}, "
        f"rtv_weight={rtv_weight}"
    )
    history_path = out_dir / "history.jsonl"
    best_val = float("inf")
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        if feature_loss is not None:
            feature_loss.train()
        running = {"total": 0.0, "reconstruction": 0.0, "feature": 0.0, "rtv": 0.0,
                   "weighted_feature": 0.0, "weighted_rtv": 0.0, "noise_sigma": 0.0}
        patch_count = 0
        progress = tqdm(train_loader, desc=f"SIDD epoch {epoch}/{epochs}")
        for batch_index, batch in enumerate(progress):
            if smoke_steps > 0 and batch_index >= smoke_steps:
                break
            noisy, gt = flatten_crops(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(noisy)
                reconstruction_loss = criterion(prediction, gt)
                raw_feature_loss = prediction.new_zeros(())
                sampled_sigma = prediction.new_zeros(())
                feature_ramp = 0.0
                if feature_loss is not None and teacher is not None:
                    visible = make_block_visible_mask(
                        noisy.shape[0], noisy.shape[2], noisy.shape[3],
                        float(config.get("feature_mask_ratio", 0.25)),
                        int(config.get("feature_mask_patch", 16)),
                        device=noisy.device, dtype=noisy.dtype, generator=mask_generator,
                    )
                    corrupted, sampled_sigmas = apply_local_gaussian_noise(
                        noisy,
                        visible,
                        float(config["feature_noise_sigma_min"]),
                        float(config["feature_noise_sigma_max"]),
                        generator=noise_generator,
                        clamp_min=0.0,
                        clamp_max=1.0,
                    )
                    bn_context = (
                        suspend_batchnorm_running_stats(model)
                        if bool(config.get("freeze_aux_bn_stats", True)) else nullcontext()
                    )
                    with bn_context:
                        _, student_features = model(corrupted, return_feats=True)
                    with torch.no_grad():
                        _, teacher_features = teacher(gt, return_feats=True)
                    raw_feature_loss, _ = feature_loss(
                        [student_features[index] for index in feature_indices],
                        [teacher_features[index] for index in feature_indices],
                        visible,
                    )
                    sampled_sigma = sampled_sigmas.mean()
                    feature_ramp = (
                        1.0 if feature_warmup_steps <= 0
                        else min(1.0, float(global_step + 1) / feature_warmup_steps)
                    )
                # RTV 的梯度比重建项更尖锐，float16 下首步可能溢出并让 GradScaler
                # 跳过 optimizer step；固定用 float32，主干前向仍保留 AMP。
                if rtv_weight > 0:
                    with torch.autocast(device_type=device.type, enabled=False):
                        raw_rtv_loss = rtv(prediction.float())
                else:
                    raw_rtv_loss = prediction.new_zeros(())
                weighted_feature = feature_weight * feature_ramp * raw_feature_loss
                weighted_rtv = rtv_weight * raw_rtv_loss
                loss = reconstruction_loss + weighted_feature + weighted_rtv
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(config.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            if teacher is not None:
                update_ema(model, teacher, float(config.get("ema_decay", 0.996)))
                teacher.eval()
            scheduler.step()
            global_step += 1
            batch_size = noisy.shape[0]
            values = {
                "total": loss,
                "reconstruction": reconstruction_loss,
                "feature": raw_feature_loss,
                "rtv": raw_rtv_loss,
                "weighted_feature": weighted_feature,
                "weighted_rtv": weighted_rtv,
                "noise_sigma": sampled_sigma,
            }
            for key, value in values.items():
                running[key] += float(value.detach()) * batch_size
            patch_count += noisy.shape[0]
            progress.set_postfix(
                rec=f"{reconstruction_loss.item():.4f}",
                feat=f"{weighted_feature.item():.4f}",
                rtv=f"{weighted_rtv.item():.4f}",
            )

        averages = {key: value / max(patch_count, 1) for key, value in running.items()}
        val_loss = validate(model, val_loader, criterion, device, amp)
        record = {
            "epoch": epoch,
            "train": averages["total"],
            "val": val_loss,
            "n2n": averages["reconstruction"],
            "mask_feature": averages["feature"],
            "rtv": averages["rtv"],
            "weighted_mask_feature": averages["weighted_feature"],
            "weighted_rtv": averages["weighted_rtv"],
            "noise_sigma": averages["noise_sigma"],
            "lr": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        update_training_curves(history_path, "SIDD supervised RGB feature ablation")
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, config, val_loss, feature_loss, teacher)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, config, val_loss, feature_loss, teacher)
        print(json.dumps(record, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scene-disjoint supervised SIDD sRGB baseline")
    parser.add_argument("--config", default="configs/sidd_supervised.json")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke_steps", type=int)
    parser.add_argument("--device", default="")
    parser.add_argument("--allow_existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
