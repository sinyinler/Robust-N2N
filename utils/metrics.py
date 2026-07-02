from __future__ import annotations

import math

import numpy as np


def center_crop_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(x: np.ndarray) -> np.ndarray:
        top = (x.shape[0] - h) // 2
        left = (x.shape[1] - w) // 2
        return x[top:top + h, left:left + w]

    return crop(a), crop(b)


def psnr(a: np.ndarray, b: np.ndarray, data_range: float | None = None) -> float:
    a, b = center_crop_to_match(np.asarray(a), np.asarray(b))
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    if data_range is None:
        data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
        if data_range <= 0:
            data_range = 1.0
    return 10.0 * math.log10((float(data_range) ** 2) / mse)


def ssim_simple(a: np.ndarray, b: np.ndarray, data_range: float | None = None) -> float:
    """轻量全图 SSIM，避免为了评估脚本额外引入依赖。"""

    a, b = center_crop_to_match(np.asarray(a), np.asarray(b))
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if data_range is None:
        data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
        if data_range <= 0:
            data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
