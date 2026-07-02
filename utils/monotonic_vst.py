from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MonotonicVSTLUT:
    y_values: np.ndarray
    f_values: np.ndarray
    path: str


def _strictly_increasing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    for idx in range(1, len(values)):
        if values[idx] <= values[idx - 1]:
            values[idx] = np.nextafter(values[idx - 1], np.float32(np.inf))
    return values


def load_vst_lut(path: str) -> MonotonicVSTLUT:
    """Load the calibrated one-dimensional VST lookup table.

    The LUT maps a raw single-pixel intensity y to f(y). Because this mapping
    is one-dimensional, it cannot use neighboring pixels and cannot secretly do
    spatial denoising.
    """
    with np.load(path, allow_pickle=False) as data:
        if "lut_y" in data and "lut_f" in data:
            # New LUTs include explicit endpoints such as 0 and 255.  Prefer
            # these points so inference does not clamp bright vessels at the
            # first/last bin center.
            y_values = np.asarray(data["lut_y"], dtype=np.float32)
            f_values = np.asarray(data["lut_f"], dtype=np.float32)
        elif "bin_centers" in data and "f" in data:
            y_values = np.asarray(data["bin_centers"], dtype=np.float32)
            f_values = np.asarray(data["f"], dtype=np.float32)
        else:
            raise ValueError(f"{path} must contain 'lut_y'/'lut_f' or 'bin_centers'/'f' arrays")

    valid = np.isfinite(y_values) & np.isfinite(f_values)
    y_values = y_values[valid]
    f_values = f_values[valid]
    if y_values.ndim != 1 or f_values.ndim != 1 or len(y_values) != len(f_values):
        raise ValueError(f"Invalid VST LUT shapes from {path}: {y_values.shape}, {f_values.shape}")
    if len(y_values) < 2:
        raise ValueError(f"VST LUT needs at least two points: {path}")

    order = np.argsort(y_values)
    y_values = y_values[order]
    f_values = f_values[order]

    y_values, unique_indices = np.unique(y_values, return_index=True)
    f_values = f_values[unique_indices]
    if len(y_values) < 2:
        raise ValueError(f"VST LUT needs at least two unique intensity points: {path}")

    f_values = np.maximum.accumulate(f_values.astype(np.float32, copy=False))
    f_values = _strictly_increasing(f_values)
    return MonotonicVSTLUT(
        y_values=y_values.astype(np.float32, copy=False),
        f_values=f_values.astype(np.float32, copy=False),
        path=path,
    )


def _as_grid(values: np.ndarray | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(values, device=ref.device, dtype=ref.dtype)


def _interp1d_torch(
    x: torch.Tensor,
    x_grid: np.ndarray | torch.Tensor,
    y_grid: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    x_grid_tensor = _as_grid(x_grid, x)
    y_grid_tensor = _as_grid(y_grid, x)
    if x_grid_tensor.numel() < 2:
        raise ValueError("Interpolation grid needs at least two points")

    x_clamped = torch.clamp(
        x,
        min=float(x_grid_tensor[0].detach().cpu()),
        max=float(x_grid_tensor[-1].detach().cpu()),
    )
    indices = torch.bucketize(x_clamped.contiguous(), x_grid_tensor.contiguous())
    indices = torch.clamp(indices, min=1, max=x_grid_tensor.numel() - 1)

    x_left = x_grid_tensor[indices - 1]
    x_right = x_grid_tensor[indices]
    y_left = y_grid_tensor[indices - 1]
    y_right = y_grid_tensor[indices]

    denom = torch.clamp(x_right - x_left, min=torch.finfo(x.dtype).eps)
    weight = (x_clamped - x_left) / denom
    return y_left + weight * (y_right - y_left)


def vst_forward_torch(
    y: torch.Tensor,
    y_values: np.ndarray | torch.Tensor,
    f_values: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    return _interp1d_torch(torch.clamp(y, min=0.0), y_values, f_values)


def vst_inverse_torch(
    z: torch.Tensor,
    y_values: np.ndarray | torch.Tensor,
    f_values: np.ndarray | torch.Tensor,
    max_value: float | None = None,
) -> torch.Tensor:
    y = _interp1d_torch(z, f_values, y_values)
    y = torch.nan_to_num(y, nan=0.0, neginf=0.0)
    if max_value is not None:
        y = torch.clamp(y, min=0.0, max=max_value)
    return torch.clamp(y, min=0.0)
