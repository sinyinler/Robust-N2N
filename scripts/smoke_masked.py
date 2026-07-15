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
from train_masked import (
    compute_gradient_diagnostics,
    suspend_batchnorm_running_stats,
    update_ema,
)


def main(device_name: str) -> None:
    device = torch.device(device_name)
    torch.manual_seed(7)
    n1 = torch.rand(2, 1, 64, 64, device=device)
    n2 = torch.rand(2, 1, 64, 64, device=device)
    mask_generator = torch.Generator(device=device).manual_seed(20_043)
    visible = make_block_visible_mask(
        2, 64, 64, 0.25, 16,
        device=device, dtype=n1.dtype, generator=mask_generator,
    )
    hidden_ratio = float((1.0 - visible).mean())
    assert visible.shape == (2, 1, 64, 64)
    assert abs(hidden_ratio - 0.25) < 1e-6, hidden_ratio

    student = MaskedDenoiserWithFeats().to(device).train()
    teacher = copy.deepcopy(student).to(device).eval().requires_grad_(False)
    masked = apply_visible_mask(n1, visible, fill="zero")

    # 独立 mask generator 必须不受全局 torch RNG 消耗影响。
    generator_a = torch.Generator(device=device).manual_seed(1234)
    generator_b = torch.Generator(device=device).manual_seed(1234)
    mask_a = make_block_visible_mask(
        2, 64, 64, 0.25, 16,
        device=device, dtype=n1.dtype, generator=generator_a,
    )
    _ = torch.rand(128, device=device)
    mask_b = make_block_visible_mask(
        2, 64, 64, 0.25, 16,
        device=device, dtype=n1.dtype, generator=generator_b,
    )
    assert torch.equal(mask_a, mask_b)

    batchnorms = [
        module for module in student.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    tracked_before = [module.num_batches_tracked.detach().clone() for module in batchnorms]
    means_before = [module.running_mean.detach().clone() for module in batchnorms]
    with suspend_batchnorm_running_stats(student):
        y_masked, student_feats = student(masked, visible, return_feats=True)
    assert all(
        torch.equal(module.num_batches_tracked, before)
        for module, before in zip(batchnorms, tracked_before)
    )
    assert all(
        torch.equal(module.running_mean, before)
        for module, before in zip(batchnorms, means_before)
    )

    y_normal = student(n1)
    assert all(
        torch.equal(module.num_batches_tracked, before + 1)
        for module, before in zip(batchnorms, tracked_before)
    )
    with torch.no_grad():
        _, teacher_feats = teacher(n2, return_feats=True)

    feature_loss = MaskedFeaturePredictionLoss([32, 64]).to(device)
    loss_pixel = masked_charbonnier(y_masked, n2, visible)
    loss_feature, per_scale = feature_loss(
        [student_feats[1], student_feats[2]],
        [teacher_feats[1], teacher_feats[2]],
        visible,
    )
    loss_n2n = (y_normal - n2).abs().mean()
    weighted_feature = 0.05 * loss_feature
    diagnostics = compute_gradient_diagnostics(
        student, loss_n2n, weighted_feature, ["encoder2", "encoder3"]
    )
    assert all(item["n2n_norm"] > 0 for item in diagnostics.values())
    assert all(item["weighted_feature_norm"] > 0 for item in diagnostics.values())
    # autograd.grad 只能返回临时张量，不能提前污染 optimizer 将使用的 parameter.grad。
    assert all(parameter.grad is None for parameter in student.parameters())

    loss = loss_n2n + loss_pixel + weighted_feature
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
        f"per_scale={[round(x, 4) for x in per_scale]} "
        f"grad_ratio_e2={diagnostics['encoder2']['feature_to_n2n_ratio']:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args().device)
