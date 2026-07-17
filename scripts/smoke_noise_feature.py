# -*- coding: utf-8 -*-
"""不依赖数据集的局部加噪 feature 分支 smoke test。"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.masked_prediction import (
    MaskedFeaturePredictionLoss,
    apply_local_gaussian_noise,
    make_block_visible_mask,
)
from models.denoiser import Denoiser
from models.denoiser_feats import DenoiserWithFeats
from train_masked import suspend_batchnorm_running_stats, update_ema
from utils.checkpoint import load_weights_flexible


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    n1 = torch.rand(2, 1, 64, 64, device=device)
    n2 = torch.rand_like(n1)

    region_generator = torch.Generator(device=device).manual_seed(20_043)
    noise_generator = torch.Generator(device=device).manual_seed(40_043)
    visible = make_block_visible_mask(
        2, 64, 64, ratio=0.25, patch=16,
        device=device, dtype=n1.dtype, generator=region_generator,
    )
    corrupted, sigmas = apply_local_gaussian_noise(
        n1, visible, 0.02, 0.06, generator=noise_generator,
    )

    hidden = 1.0 - visible
    assert torch.equal(corrupted * visible, n1 * visible), "未选中区域不应被改变"
    assert float(((corrupted - n1).abs() * hidden).sum()) > 0.0
    assert bool(((sigmas >= 0.02) & (sigmas <= 0.06)).all())

    # Gaussian 模式必须是单通道；region map 不传入 student。
    student = DenoiserWithFeats(input_channels=1).to(device).train()
    teacher = DenoiserWithFeats(input_channels=1).to(device).eval()
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    predictor = MaskedFeaturePredictionLoss([32, 64]).to(device)

    with suspend_batchnorm_running_stats(student):
        _, noisy_feats = student(corrupted, return_feats=True)
    with torch.no_grad():
        _, target_feats = teacher(n2, return_feats=True)
    feature_loss, per_scale = predictor(
        [noisy_feats[1], noisy_feats[2]],
        [target_feats[1], target_feats[2]],
        visible,
    )
    feature_loss.backward()
    assert torch.isfinite(feature_loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in per_scale)
    assert any(parameter.grad is not None for parameter in student.parameters())
    update_ema(student, teacher, decay=0.996)

    # 训练 checkpoint 的 backbone 与原始单通道 N2N 推理结构完全兼容。
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "noise_feature.pth"
        torch.save({"model": student.state_dict()}, checkpoint)
        inference_model = Denoiser(input_channels=1).to(device)
        loaded = load_weights_flexible(inference_model, str(checkpoint), device)
    assert loaded["skipped"] == 0, loaded
    assert loaded["loaded"] == len(inference_model.state_dict()), loaded

    print(
        f"[OK] local Gaussian feature smoke device={device} "
        f"hidden={float(hidden.mean()):.3f} "
        f"sigma={float(sigmas.mean()):.4f} feature={float(feature_loss.detach()):.4f} "
        f"checkpoint={loaded}"
    )


if __name__ == "__main__":
    main()
