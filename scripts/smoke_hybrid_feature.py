# -*- coding: utf-8 -*-
"""Data-free forward/backward smoke test for both hybrid corruption modes."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.masked_prediction import (
    MaskedFeaturePredictionLoss,
    apply_local_hybrid_noise,
    make_block_visible_mask,
)
from models.denoiser_feats import DenoiserWithFeats
from train_masked import suspend_batchnorm_running_stats


def run_strategy(strategy: str, device: torch.device) -> None:
    torch.manual_seed(42)
    n1 = torch.rand(2, 1, 64, 64, device=device)
    n2 = torch.rand_like(n1)
    visible = make_block_visible_mask(
        2,
        64,
        64,
        ratio=0.25,
        patch=8,
        device=device,
        dtype=n1.dtype,
        generator=torch.Generator(device=device).manual_seed(20_043),
    )
    strength = 1.0 if strategy == "mixture" else 2.0 ** -0.5
    corrupted, cvs, sigmas, gamma_fraction = apply_local_hybrid_noise(
        n1,
        visible,
        patch=8,
        gamma_probability=0.5,
        gamma_cv_min=0.025 * strength,
        gamma_cv_max=0.075 * strength,
        gaussian_sigma_min=0.02 * strength,
        gaussian_sigma_max=0.06 * strength,
        strategy=strategy,
        gamma_generator=torch.Generator(device=device).manual_seed(40_043),
        gaussian_generator=torch.Generator(device=device).manual_seed(50_043),
        choice_generator=torch.Generator(device=device).manual_seed(60_043),
    )

    student = DenoiserWithFeats(input_channels=1).to(device).train()
    teacher = DenoiserWithFeats(input_channels=1).to(device).eval()
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    feature_loss = MaskedFeaturePredictionLoss([32, 64]).to(device)

    with suspend_batchnorm_running_stats(student):
        _, student_feats = student(corrupted, return_feats=True)
    with torch.no_grad():
        _, teacher_feats = teacher(n2, return_feats=True)
    loss, _ = feature_loss(
        [student_feats[1], student_feats[2]],
        [teacher_feats[1], teacher_feats[2]],
        visible,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.equal(corrupted * visible, n1 * visible)
    assert any(parameter.grad is not None for parameter in student.parameters())
    print(
        f"[OK] strategy={strategy} device={device} loss={float(loss.detach()):.5f} "
        f"gamma_cv={float(cvs.mean()):.5f} sigma={float(sigmas.mean()):.5f} "
        f"gamma_fraction={float(gamma_fraction.mean()):.3f}"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_strategy("mixture", device)
    run_strategy("sequential", device)


if __name__ == "__main__":
    main()
