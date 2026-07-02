from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.monotonic_vst import load_vst_lut, vst_forward_torch, vst_inverse_torch


BOXCOX_LAM = -0.15
BOXCOX_EPS = 1e-6
LAMBDA_MIN = -0.3
LAMBDA_MAX = 0.2


@dataclass
class IntensityTransform:
    """统一管理训练和推理里的强度变换。

    NTN 的噪声翻译目标是 Gaussian noise。对 LSCI/BFI 这类信号相关噪声，
    先进入 VST 域可以让“翻译成高斯噪声”这件事更接近论文假设。
    """

    name: str = "log1p"
    boxcox_lam: float = BOXCOX_LAM
    boxcox_eps: float = BOXCOX_EPS
    vst_lut: str = ""

    def __post_init__(self):
        self.name = self.name.lower()
        if self.name not in {"log1p", "boxcox", "learned_vst", "none"}:
            raise ValueError("intensity transform must be log1p, boxcox, learned_vst, or none")
        self._lut = None
        if self.name == "learned_vst":
            if not self.vst_lut:
                raise ValueError("vst_lut is required when using learned_vst")
            self._lut = load_vst_lut(self.vst_lut)

    def forward(self, x: torch.Tensor, lam: float | None = None) -> torch.Tensor:
        if self.name == "none":
            return x
        if self.name == "log1p":
            return torch.log1p(torch.clamp(x, min=0.0))
        if self.name == "learned_vst":
            return vst_forward_torch(x, self._lut.y_values, self._lut.f_values)

        value = self.boxcox_lam if lam is None else float(lam)
        u = torch.clamp(x, min=0.0) + 1.0 + self.boxcox_eps
        if abs(value) < 1e-12:
            return torch.log(u)
        return (torch.pow(u, value) - 1.0) / value

    def inverse(self, z: torch.Tensor, max_value: float | None = None, lam: float | None = None) -> torch.Tensor:
        if self.name == "none":
            out = torch.nan_to_num(z, nan=0.0, neginf=0.0)
            return torch.clamp(out, min=0.0, max=max_value) if max_value is not None else torch.clamp(out, min=0.0)
        if self.name == "log1p":
            out = torch.expm1(z)
            out = torch.nan_to_num(out, nan=0.0, neginf=0.0)
            return torch.clamp(out, min=0.0, max=max_value) if max_value is not None else torch.clamp(out, min=0.0)
        if self.name == "learned_vst":
            return vst_inverse_torch(z, self._lut.y_values, self._lut.f_values, max_value=max_value)

        value = self.boxcox_lam if lam is None else float(lam)
        if abs(value) < 1e-12:
            out = torch.expm1(z)
        else:
            if value < 0:
                z = torch.clamp(z, max=(-1.0 / value) - self.boxcox_eps)
            u = torch.clamp(value * z + 1.0, min=self.boxcox_eps)
            out = torch.pow(u, 1.0 / value) - 1.0
        out = torch.nan_to_num(out, nan=0.0, neginf=0.0)
        return torch.clamp(out, min=0.0, max=max_value) if max_value is not None else torch.clamp(out, min=0.0)


def lambda_condition_value(lam: float, lambda_min: float = LAMBDA_MIN, lambda_max: float = LAMBDA_MAX) -> float:
    if abs(lambda_max - lambda_min) < 1e-12:
        return float(lam)
    return float(2.0 * (lam - lambda_min) / (lambda_max - lambda_min) - 1.0)


def append_condition_channel(x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
    """给 boxcox lambda-conditioned 模型拼接条件通道。"""

    if condition is None or condition.numel() == 0:
        return x
    if condition.ndim == 2:
        condition = condition.unsqueeze(0).unsqueeze(0)
    elif condition.ndim == 3:
        condition = condition.unsqueeze(1)
    return torch.cat((x, condition.to(device=x.device, dtype=x.dtype)), dim=1)
