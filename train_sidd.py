"""SIDD-Small sRGB 的 scene-disjoint 监督训练入口。"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.sidd_dataset import SIDDSceneCropDataset, seed_sidd_worker
from losses.charbonnier import CharbonnierLoss
from models.sidd_rgb_denoiser import SIDDRGBDenoiser
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


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "val_loss": val_loss,
        },
        path,
    )


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
    optimizer = AdamW(
        model.parameters(), lr=float(config["lr_max"]), weight_decay=float(config["weight_decay"])
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

    print(
        f"[SIDD] device={device}, train_pairs={len(train_dataset.pairs)}, "
        f"val_pairs={len(val_dataset.pairs)}, effective_patch_batch="
        f"{config['image_batch_size'] * config['crops_per_load']}, params="
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    history_path = out_dir / "history.jsonl"
    best_val = float("inf")
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        patch_count = 0
        progress = tqdm(train_loader, desc=f"SIDD epoch {epoch}/{epochs}")
        for batch_index, batch in enumerate(progress):
            if smoke_steps > 0 and batch_index >= smoke_steps:
                break
            noisy, gt = flatten_crops(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(noisy)
                loss = criterion(prediction, gt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            running += float(loss.item()) * noisy.shape[0]
            patch_count += noisy.shape[0]
            progress.set_postfix(loss=f"{loss.item():.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = running / max(patch_count, 1)
        val_loss = validate(model, val_loader, criterion, device, amp)
        record = {
            "epoch": epoch,
            "train": train_loss,
            "val": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        update_training_curves(history_path, "SIDD supervised RGB Charbonnier")
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, config, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, config, val_loss)
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
