# -*- coding: utf-8 -*-
"""Masked reconstruction and masked feature-prediction objectives."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def make_block_visible_mask(
    batch: int,
    height: int,
    width: int,
    ratio: float,
    patch: int,
    *,
    device,
    dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return ``(N,1,H,W)`` with 1=visible and 0=hidden block patches.

    Every sample hides the same *number* of grid cells but draws independent
    locations.  Border cells are cropped when H/W are not multiples of patch.
    """
    ratio = float(ratio)
    patch = int(patch)
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"mask ratio must be in [0,1), got {ratio}")
    if patch <= 0:
        raise ValueError(f"mask patch must be positive, got {patch}")
    if ratio == 0.0:
        return torch.ones((batch, 1, height, width), device=device, dtype=dtype)

    gh = math.ceil(height / patch)
    gw = math.ceil(width / patch)
    cells = gh * gw
    hidden_cells = min(cells - 1, max(1, int(round(cells * ratio))))
    # 使用独立 generator，避免 predictor 初始化或 DataLoader shuffle 改变 mask 序列。
    scores = torch.rand((batch, cells), device=device, generator=generator)
    hidden_idx = scores.topk(hidden_cells, dim=1, largest=True, sorted=False).indices
    hidden_grid = torch.zeros((batch, cells), device=device, dtype=dtype)
    hidden_grid.scatter_(1, hidden_idx, 1.0)
    hidden_grid = hidden_grid.view(batch, 1, gh, gw)
    hidden = hidden_grid.repeat_interleave(patch, dim=2).repeat_interleave(patch, dim=3)
    hidden = hidden[..., :height, :width]
    return 1.0 - hidden


def apply_visible_mask(
    image: torch.Tensor,
    visible_mask: torch.Tensor,
    fill: str = "zero",
) -> torch.Tensor:
    """Hide image pixels while keeping mask creation independent of content."""
    if fill == "zero":
        fill_value = torch.zeros_like(image)
    elif fill == "mean":
        fill_value = image.mean(dim=(2, 3), keepdim=True).expand_as(image)
    else:
        raise ValueError(f"unsupported mask fill {fill!r}; choose zero or mean")
    return image * visible_mask + fill_value * (1.0 - visible_mask)


def masked_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier reconstruction averaged only over hidden image pixels."""
    hidden = 1.0 - visible_mask.to(device=prediction.device, dtype=prediction.dtype)
    if hidden.shape[1] == 1 and prediction.shape[1] != 1:
        hidden = hidden.expand(-1, prediction.shape[1], -1, -1)
    error = torch.sqrt((prediction - target) ** 2 + float(eps) ** 2)
    return (error * hidden).sum() / hidden.sum().clamp_min(1.0)


class FeaturePredictor(nn.Module):
    """Per-location predictor without BatchNorm spatial-statistic leakage."""

    def __init__(self, channels: int, hidden_ratio: float = 1.0):
        super().__init__()
        hidden = max(8, int(round(channels * float(hidden_ratio))))
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedFeaturePredictionLoss(nn.Module):
    """Predict full-view teacher features at locations hidden from the student."""

    def __init__(self, channels, weights=None, predictor_hidden_ratio: float = 1.0):
        super().__init__()
        channels = list(channels)
        if not channels:
            raise ValueError("at least one feature scale is required")
        self.predictors = nn.ModuleList(
            [FeaturePredictor(c, hidden_ratio=predictor_hidden_ratio) for c in channels]
        )
        raw_weights = list(weights) if weights is not None else [1.0] * len(channels)
        if (len(raw_weights) != len(channels) or any(w < 0 for w in raw_weights)
                or sum(raw_weights) <= 0):
            raise ValueError("feature weights must match channels and have a positive sum")
        total = float(sum(raw_weights))
        self.weights = [float(w) / total for w in raw_weights]

    def forward(self, student_feats, teacher_feats, visible_mask: torch.Tensor):
        if len(student_feats) != len(self.predictors) or len(teacher_feats) != len(self.predictors):
            raise ValueError("student/teacher feature counts must match configured predictors")
        total = student_feats[0].new_zeros(())
        per_scale = []
        hidden_image = 1.0 - visible_mask
        for student, teacher, predictor, weight in zip(
            student_feats, teacher_feats, self.predictors, self.weights
        ):
            if student.shape != teacher.shape:
                raise ValueError(
                    f"student feature {tuple(student.shape)} != teacher feature {tuple(teacher.shape)}"
                )
            hidden = F.interpolate(hidden_image, size=student.shape[-2:], mode="nearest")
            predicted = F.normalize(predictor(student), dim=1)
            target = F.normalize(teacher.detach(), dim=1)
            distance = 1.0 - (predicted * target).sum(dim=1, keepdim=True)
            value = (distance * hidden).sum() / hidden.sum().clamp_min(1.0)
            total = total + float(weight) * value
            per_scale.append(float(value.detach()))
        return total, per_scale


__all__ = [
    "make_block_visible_mask",
    "apply_visible_mask",
    "masked_charbonnier",
    "MaskedFeaturePredictionLoss",
]
