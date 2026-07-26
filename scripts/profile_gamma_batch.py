# -*- coding: utf-8 -*-
"""在单张 GPU 上复现 Gamma feature 训练步，探测安全 micro-batch。"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.charbonnier import CharbonnierLoss
from losses.masked_prediction import (
    MaskedFeaturePredictionLoss,
    apply_local_gamma_noise,
    make_block_visible_mask,
)
from losses.rtv import RTVRegularizer
from models.denoiser_feats import DenoiserWithFeats
from train_masked import (
    compute_gradient_diagnostics,
    suspend_batchnorm_running_stats,
    update_ema,
)


def gib(value: int) -> float:
    return float(value) / 2**30


def profile(batch: int, crop: int, seed: int, diagnostics: bool) -> dict[str, float]:
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    student = DenoiserWithFeats(input_channels=1).to(device).train()
    teacher = DenoiserWithFeats(input_channels=1).to(device).eval()
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    predictor = MaskedFeaturePredictionLoss([32, 64]).to(device).train()
    trainable = list(student.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=0.01, weight_decay=1e-4)
    charb = CharbonnierLoss(eps=1e-3).to(device)
    rtv = RTVRegularizer(radius=2, sigma=2.0, eps=1e-3).to(device)

    mask_generator = torch.Generator(device=device).manual_seed(seed + 20_001)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 40_001)
    for step in range(2):
        # 用真实 Level4 的典型 raw 强度范围生成 log1p 输入。
        n1 = torch.log1p(torch.rand(batch, 1, crop, crop, device=device) * 300.0)
        n2 = torch.log1p(torch.rand_like(n1) * 300.0)
        visible = make_block_visible_mask(
            batch, crop, crop, ratio=0.25, patch=16,
            device=device, dtype=n1.dtype, generator=mask_generator,
        )
        corrupted, _ = apply_local_gamma_noise(
            n1, visible, 0.025, 0.075, generator=noise_generator,
        )
        with suspend_batchnorm_running_stats(student):
            _, noisy_feats = student(corrupted, return_feats=True)
        with torch.no_grad():
            _, target_feats = teacher(n2, return_feats=True)
        feature, _ = predictor(
            [noisy_feats[1], noisy_feats[2]],
            [target_feats[1], target_feats[2]],
            visible,
        )
        normal = student(n1)
        n2n = charb(normal, n2)
        total = n2n + 0.01 * rtv(normal) + 0.10 * feature

        if diagnostics and step == 0:
            compute_gradient_diagnostics(
                student, n2n, 0.10 * feature, ["encoder2", "encoder3"]
            )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        update_ema(student, teacher, decay=0.996)

    torch.cuda.synchronize(device)
    result = {
        "allocated_gib": gib(torch.cuda.memory_allocated(device)),
        "reserved_gib": gib(torch.cuda.memory_reserved(device)),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
    }
    del optimizer, predictor, teacher, student, n1, n2, visible, corrupted
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile local Gamma feature batch memory")
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 6, 4])
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--seed", type=int, default=187)
    parser.add_argument("--diagnostics", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for batch profiling")
    print(
        f"[INFO] gpu={torch.cuda.get_device_name(0)} "
        f"total={gib(torch.cuda.get_device_properties(0).total_memory):.2f} GiB "
        f"crop={args.crop} seed={args.seed}"
    )
    for batch in args.batches:
        try:
            stats = profile(batch, args.crop, args.seed, bool(args.diagnostics))
        except torch.OutOfMemoryError as error:
            print(f"[OOM] batch={batch}: {error}")
            gc.collect()
            torch.cuda.empty_cache()
            continue
        fields = " ".join(f"{key}={value:.3f}" for key, value in stats.items())
        print(f"[OK] batch={batch} {fields}")
        break
    else:
        raise RuntimeError("all requested batch sizes ran out of GPU memory")


if __name__ == "__main__":
    main()
