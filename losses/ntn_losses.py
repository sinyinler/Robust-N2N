from __future__ import annotations

import torch
from torch import nn


def sorted_wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """用排序后的 L1 近似一维 1-Wasserstein distance。"""

    a_sorted, _ = torch.sort(a, dim=-1)
    b_sorted, _ = torch.sort(b, dim=-1)
    return torch.mean(torch.abs(a_sorted - b_sorted))


def spatial_gaussian_loss(noise: torch.Tensor) -> torch.Tensor:
    """约束 translated noise 的逐像素分布接近 Gaussian。"""

    b, c = noise.shape[:2]
    flat = noise.reshape(b, c, -1)
    mean = flat.mean(dim=-1, keepdim=True)
    std = flat.std(dim=-1, keepdim=True).clamp_min(1e-6)
    gaussian = torch.randn_like(flat) * std + mean
    return sorted_wasserstein_1d(flat, gaussian)


def frequency_rayleigh_loss(noise: torch.Tensor, highpass_ratio: float = 0.0) -> torch.Tensor:
    """约束 translated noise 的频域幅值接近白 Gaussian 对应的 Rayleigh 分布。

    highpass_ratio > 0 时，只在高频区域计算，可用于保护血管等低频/中频结构。
    """

    b, c, h, w = noise.shape
    flat = noise.reshape(b, c, -1)
    mean = flat.mean(dim=-1, keepdim=True)
    std = flat.std(dim=-1, keepdim=True).clamp_min(1e-6)
    gaussian = (torch.randn_like(flat) * std + mean).reshape(b, c, h, w)

    ft = torch.fft.fft2(noise).abs()
    fg = torch.fft.fft2(gaussian).abs()

    if highpass_ratio > 0:
        yy = torch.fft.fftfreq(h, device=noise.device, dtype=noise.dtype).view(h, 1)
        xx = torch.fft.fftfreq(w, device=noise.device, dtype=noise.dtype).view(1, w)
        radius = torch.sqrt(xx * xx + yy * yy)
        mask = radius >= float(highpass_ratio)
        ft = ft[..., mask]
        fg = fg[..., mask]
    else:
        ft = ft.reshape(b, c, -1)
        fg = fg.reshape(b, c, -1)

    return sorted_wasserstein_1d(ft, fg)


class ExplicitNoiseTranslationLoss(nn.Module):
    """论文中的 explicit loss: Lspatial + beta * Lfreq。"""

    def __init__(self, beta: float = 2e-3, highpass_ratio: float = 0.0):
        super().__init__()
        self.beta = float(beta)
        self.highpass_ratio = float(highpass_ratio)

    def forward(self, translated_noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = spatial_gaussian_loss(translated_noise)
        freq = frequency_rayleigh_loss(translated_noise, highpass_ratio=self.highpass_ratio)
        total = spatial + self.beta * freq
        return total, spatial, freq
