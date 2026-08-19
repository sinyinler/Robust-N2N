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


def apply_local_gaussian_noise(
    image: torch.Tensor,
    visible_mask: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    *,
    generator: torch.Generator | None = None,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只在选中区域加入独立 Gaussian noise，并返回每张图实际采样的 sigma。

    ``visible_mask`` 沿用现有约定：1 表示未扰动，0 表示被选中的区域。它只负责
    圈定扰动和 feature loss，不需要作为网络输入。每张图从
    ``[sigma_min, sigma_max]`` 独立均匀采样一个噪声强度。
    """
    sigma_min = float(sigma_min)
    sigma_max = float(sigma_max)
    if sigma_min < 0.0 or sigma_max < sigma_min:
        raise ValueError(
            f"noise sigma range must satisfy 0 <= min <= max, got {sigma_min}..{sigma_max}"
        )
    if image.ndim != 4:
        raise ValueError(f"image must have shape (N,C,H,W), got {tuple(image.shape)}")
    if visible_mask.ndim != 4 or visible_mask.shape[1] != 1:
        raise ValueError(
            f"visible_mask must have shape (N,1,H,W), got {tuple(visible_mask.shape)}"
        )
    if visible_mask.shape[0] != image.shape[0] or visible_mask.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"visible_mask shape {tuple(visible_mask.shape)} is incompatible with image "
            f"{tuple(image.shape)}"
        )

    visible = visible_mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
    hidden = 1.0 - visible
    if image.shape[1] != 1:
        hidden = hidden.expand(-1, image.shape[1], -1, -1)

    unit = torch.rand(
        (image.shape[0], 1, 1, 1),
        device=image.device,
        dtype=image.dtype,
        generator=generator,
    )
    sigmas = sigma_min + (sigma_max - sigma_min) * unit
    noise = torch.randn(
        image.shape,
        device=image.device,
        dtype=image.dtype,
        generator=generator,
    ) * sigmas
    corrupted = image + hidden * noise
    if clamp_min is not None:
        corrupted = corrupted.clamp_min(float(clamp_min))
    if clamp_max is not None:
        corrupted = corrupted.clamp_max(float(clamp_max))
    return corrupted, sigmas


def apply_local_gamma_noise(
    log_image: torch.Tensor,
    visible_mask: torch.Tensor,
    cv_min: float,
    cv_max: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只在选中区域施加 raw 域 Gamma 乘性噪声。

    输入必须是 ``log1p`` 域的非负图像。每张图独立采样
    ``CV ~ Uniform(cv_min, cv_max)``，再令
    ``factor ~ Gamma(k, rate=k)``、``k=1/CV^2``。因此 factor 的均值为 1，
    raw 域扰动在期望上不改变亮度。未选中区域直接复用原张量，避免
    ``expm1 -> log1p`` 往返产生数值差异。
    """
    cv_min = float(cv_min)
    cv_max = float(cv_max)
    if cv_min <= 0.0 or cv_max < cv_min:
        raise ValueError(
            f"gamma CV range must satisfy 0 < min <= max, got {cv_min}..{cv_max}"
        )
    if log_image.ndim != 4:
        raise ValueError(
            f"log_image must have shape (N,C,H,W), got {tuple(log_image.shape)}"
        )
    if visible_mask.ndim != 4 or visible_mask.shape[1] != 1:
        raise ValueError(
            f"visible_mask must have shape (N,1,H,W), got {tuple(visible_mask.shape)}"
        )
    if (
        visible_mask.shape[0] != log_image.shape[0]
        or visible_mask.shape[-2:] != log_image.shape[-2:]
    ):
        raise ValueError(
            f"visible_mask shape {tuple(visible_mask.shape)} is incompatible with "
            f"log_image {tuple(log_image.shape)}"
        )
    visible = visible_mask.to(device=log_image.device, dtype=log_image.dtype).clamp(0.0, 1.0)
    hidden = 1.0 - visible
    if log_image.shape[1] != 1:
        visible = visible.expand(-1, log_image.shape[1], -1, -1)
        hidden = hidden.expand(-1, log_image.shape[1], -1, -1)

    unit = torch.rand(
        (log_image.shape[0], 1, 1, 1),
        device=log_image.device,
        dtype=log_image.dtype,
        generator=generator,
    )
    cvs = cv_min + (cv_max - cv_min) * unit
    shapes = cvs.reciprocal().square()
    expanded_shapes = shapes.expand_as(log_image)

    # torch.distributions.Gamma 不接收独立 generator；底层标准 Gamma sampler
    # 支持 generator，可保持 region/DataLoader/corruption 三条随机轨迹相互隔离。
    factors = torch._standard_gamma(expanded_shapes, generator=generator) / expanded_shapes
    raw = torch.expm1(log_image)
    corrupted_hidden = torch.log1p(raw * factors)
    corrupted = log_image * visible + corrupted_hidden * hidden
    return corrupted, cvs


def split_hidden_regions_by_patch(
    visible_mask: torch.Tensor,
    patch: int,
    gamma_probability: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split selected patches into mutually exclusive Gamma/Gaussian regions.

    ``visible_mask`` follows the project convention: one means unperturbed and
    zero means selected for the auxiliary feature task. Each selected patch is
    assigned to Gamma with ``gamma_probability`` and to Gaussian otherwise.
    The two returned hidden regions are disjoint and their union exactly equals
    the original hidden region.
    """
    patch = int(patch)
    gamma_probability = float(gamma_probability)
    if patch <= 0:
        raise ValueError(f"patch must be positive, got {patch}")
    if not 0.0 <= gamma_probability <= 1.0:
        raise ValueError(
            "gamma_probability must be in [0,1], "
            f"got {gamma_probability}"
        )
    if visible_mask.ndim != 4 or visible_mask.shape[1] != 1:
        raise ValueError(
            "visible_mask must have shape (N,1,H,W), "
            f"got {tuple(visible_mask.shape)}"
        )

    batch, _, height, width = visible_mask.shape
    gh = math.ceil(height / patch)
    gw = math.ceil(width / patch)
    scores = torch.rand(
        (batch, gh * gw),
        device=visible_mask.device,
        generator=generator,
    )

    # The incoming region mask is block-constant. Sample exactly the requested
    # fraction among its selected grid cells instead of using Bernoulli draws,
    # so the 50/50 experiment has a fixed corruption budget per image.
    hidden_grid = (1.0 - visible_mask[:, :, ::patch, ::patch]).reshape(batch, -1)
    assignment_grid = torch.zeros_like(hidden_grid)
    for sample_index in range(batch):
        selected = torch.nonzero(hidden_grid[sample_index] > 0.5, as_tuple=False).flatten()
        gamma_cells = int(round(selected.numel() * gamma_probability))
        if gamma_cells > 0:
            order = scores[sample_index, selected].topk(
                gamma_cells, largest=True, sorted=False
            ).indices
            assignment_grid[sample_index, selected[order]] = 1.0
    assignment = assignment_grid.view(batch, 1, gh, gw)
    assignment = assignment.repeat_interleave(patch, dim=2).repeat_interleave(
        patch, dim=3
    )[..., :height, :width]

    visible = visible_mask.clamp(0.0, 1.0)
    hidden = 1.0 - visible
    gamma_hidden = hidden * assignment
    gaussian_hidden = hidden * (1.0 - assignment)
    gamma_visible = 1.0 - gamma_hidden
    gaussian_visible = 1.0 - gaussian_hidden

    hidden_count = hidden.flatten(1).sum(dim=1).clamp_min(1.0)
    gamma_fraction = gamma_hidden.flatten(1).sum(dim=1) / hidden_count
    return gamma_visible, gaussian_visible, gamma_fraction


def apply_local_hybrid_noise(
    log_image: torch.Tensor,
    visible_mask: torch.Tensor,
    patch: int,
    gamma_probability: float,
    gamma_cv_min: float,
    gamma_cv_max: float,
    gaussian_sigma_min: float,
    gaussian_sigma_max: float,
    *,
    strategy: str,
    gamma_generator: torch.Generator | None = None,
    gaussian_generator: torch.Generator | None = None,
    choice_generator: torch.Generator | None = None,
    clamp_min: float | None = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply complementary raw-Gamma and log-Gaussian perturbations.

    ``strategy='mixture'`` assigns each selected patch to exactly one noise
    family. ``strategy='sequential'`` applies raw-domain Gamma first and then
    log-domain Gaussian to every selected patch. The caller controls the two
    strengths explicitly, so reduced sequential ranges are recorded directly
    in the run configuration.
    """
    if strategy == "mixture":
        gamma_visible, gaussian_visible, gamma_fraction = (
            split_hidden_regions_by_patch(
                visible_mask,
                patch,
                gamma_probability,
                generator=choice_generator,
            )
        )
    elif strategy == "sequential":
        gamma_visible = visible_mask
        gaussian_visible = visible_mask
        gamma_fraction = torch.ones(
            (log_image.shape[0],),
            device=log_image.device,
            dtype=log_image.dtype,
        )
    else:
        raise ValueError(
            f"unsupported hybrid strategy {strategy!r}; "
            "choose 'mixture' or 'sequential'"
        )

    gamma_corrupted, cvs = apply_local_gamma_noise(
        log_image,
        gamma_visible,
        gamma_cv_min,
        gamma_cv_max,
        generator=gamma_generator,
    )
    corrupted, sigmas = apply_local_gaussian_noise(
        gamma_corrupted,
        gaussian_visible,
        gaussian_sigma_min,
        gaussian_sigma_max,
        generator=gaussian_generator,
        clamp_min=clamp_min,
    )
    return corrupted, cvs, sigmas, gamma_fraction


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
    "apply_local_gaussian_noise",
    "apply_local_gamma_noise",
    "apply_local_hybrid_noise",
    "masked_charbonnier",
    "MaskedFeaturePredictionLoss",
    "split_hidden_regions_by_patch",
]
