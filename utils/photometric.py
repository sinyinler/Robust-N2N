# -*- coding: utf-8 -*-
"""Reference-based photometric diagnostics.

These helpers deliberately use the reference image to fit an affine mapping.
They are therefore suitable for error diagnosis only, not deployable inference
or headline benchmark metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffinePhotometricFit:
    """Least-squares fit of ``reference ~= scale * output + offset``."""

    corrected: np.ndarray
    scale: float
    offset: float
    output_mean: float
    reference_mean: float
    output_std: float
    reference_std: float
    mean_ratio: float
    std_ratio: float


def _center_crop(a: np.ndarray, height: int, width: int) -> np.ndarray:
    top = max(0, (a.shape[0] - height) // 2)
    left = max(0, (a.shape[1] - width) // 2)
    return a[top:top + height, left:left + width]


def common_center_crop(
    output: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop two 2-D arrays to their largest common spatial shape."""
    output = np.squeeze(np.asarray(output))
    reference = np.squeeze(np.asarray(reference))
    if output.ndim != 2 or reference.ndim != 2:
        raise ValueError(
            "Photometric fitting expects two 2-D images, got "
            f"{output.shape} and {reference.shape}"
        )
    height = min(output.shape[0], reference.shape[0])
    width = min(output.shape[1], reference.shape[1])
    return _center_crop(output, height, width), _center_crop(reference, height, width)


def fit_reference_affine(
    output: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-12,
) -> AffinePhotometricFit:
    """Fit ``reference ~= scale * output + offset`` by least squares.

    The fit is computed over all finite pixels in the common center crop. The
    returned correction is applied to the full output image without clipping.
    In the degenerate constant-output case, the best constant prediction is the
    reference mean (``scale=0``).
    """
    output_array = np.asarray(output, dtype=np.float64)
    cropped_output, cropped_reference = common_center_crop(output_array, reference)
    cropped_output = cropped_output.astype(np.float64, copy=False)
    cropped_reference = cropped_reference.astype(np.float64, copy=False)

    finite = np.isfinite(cropped_output) & np.isfinite(cropped_reference)
    if not finite.any():
        raise ValueError("Photometric fitting found no finite pixel pairs")

    y = cropped_output[finite]
    target = cropped_reference[finite]
    output_mean = float(y.mean())
    reference_mean = float(target.mean())
    output_std = float(y.std())
    reference_std = float(target.std())

    centered_y = y - output_mean
    denominator = float(np.dot(centered_y, centered_y))
    tolerance = eps * max(1.0, float(np.dot(y, y)))
    if denominator <= tolerance:
        scale = 0.0
        offset = reference_mean
    else:
        scale = float(np.dot(centered_y, target - reference_mean) / denominator)
        offset = float(reference_mean - scale * output_mean)

    mean_ratio = (
        float(output_mean / reference_mean)
        if abs(reference_mean) > eps else float("nan")
    )
    std_ratio = (
        float(output_std / reference_std)
        if reference_std > eps else float("nan")
    )
    corrected = scale * output_array + offset
    return AffinePhotometricFit(
        corrected=corrected,
        scale=scale,
        offset=offset,
        output_mean=output_mean,
        reference_mean=reference_mean,
        output_std=output_std,
        reference_std=reference_std,
        mean_ratio=mean_ratio,
        std_ratio=std_ratio,
    )
