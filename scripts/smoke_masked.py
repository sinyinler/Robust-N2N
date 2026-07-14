# -*- coding: utf-8 -*-
"""CPU/GPU smoke test for the masked training path (no dataset required)."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.masked_denoiser import MaskedDenoiserWithFeats
from losses.masked_prediction import (
    MaskedFeaturePredictionLoss,
    apply_visible_mask,
    make_block_visible_mask,
    masked_charbonnier,
)
from train_masked import update_ema


def main(device_name: str) -> None:
    device = torch.device(device_name)
    torch.manual_seed(7)
    n1 = torch.rand(2, 1, 64, 64, device=device)
    n2 = torch.rand(2, 1, 64, 64, device=device)
    visible = make_block_visible_mask(2, 64, 64, 0.25, 16, device=device, dtype=n1.dtype)
    hidden_ratio = float((1.0 - visible).mean())
    assert visible.shape == (2, 1, 64, 64)
    assert abs(hidden_ratio - 0.25) < 1e-6, hidden_ratio

    student = MaskedDenoiserWithFeats().to(device).train()
    teacher = copy.deepcopy(student).to(device).eval().requires_grad_(False)
    masked = apply_visible_mask(n1, visible, fill="zero")
    y_normal = student(n1)
    y_masked, student_feats = student(masked, visible, return_feats=True)
    with torch.no_grad():
        _, teacher_feats = teacher(n2, return_feats=True)

    feature_loss = MaskedFeaturePredictionLoss([32, 64]).to(device)
    loss_pixel = masked_charbonnier(y_masked, n2, visible)
    loss_feature, per_scale = feature_loss(
        [student_feats[1], student_feats[2]],
        [teacher_feats[1], teacher_feats[2]],
        visible,
    )
    loss = (y_normal - n2).abs().mean() + loss_pixel + 0.05 * loss_feature
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())

    before = next(teacher.parameters()).detach().clone()
    with torch.no_grad():
        next(student.parameters()).add_(0.1)
    update_ema(student, teacher, 0.9)
    after = next(teacher.parameters()).detach()
    assert not torch.equal(before, after)
    assert y_normal.shape == n1.shape == y_masked.shape
    print(
        f"[OK] masked smoke device={device} hidden={hidden_ratio:.3f} "
        f"pixel={float(loss_pixel):.4f} feature={float(loss_feature):.4f} "
        f"per_scale={[round(x, 4) for x in per_scale]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args().device)
