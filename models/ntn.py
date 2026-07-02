from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """轻量 NAF-style block。

    论文 NTN 使用 NAFBlock 作为 GIBlock 的基础块。这里保留无显式激活、
    depthwise convolution、simple gate 与 channel attention 这些核心设计。
    """

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channels = channels * dw_expand
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            padding_mode="reflect",
            bias=True,
        )
        self.sg = SimpleGate()
        self.sca = ChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)

        ffn_channels = channels * ffn_expand
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.ffn2 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)

        # 从近似恒等映射开始，避免初期破坏输入信号。
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + self.beta * y

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg(y)
        y = self.ffn2(y)
        return x + self.gamma * y


class GaussianInjectionBlock(nn.Module):
    """GIBlock: NAFBlock 后注入可学习强度的 Gaussian noise。"""

    def __init__(self, channels: int, inject_sigma: float = 1.0, init_noise_scale: float = 0.1):
        super().__init__()
        self.naf = NAFBlock(channels)
        self.inject_sigma = float(inject_sigma)
        # 论文强调 Gaussian 注入是 NTN 的关键设计。noise_scale 用小正值初始化（而非 0），
        # 让 GIBlock 从训练第一步就真正注入高斯先验，可学习强度后续再自适应调整。
        self.noise_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(init_noise_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.naf(x)
        # Robust-N2N：Gaussian 注入只在训练时作为随机正则启用；推理时(self.training=False)关闭，
        # 保证输出确定、可复现。（NTN 原项目未做此 gating，推理时仍在注入随机噪声。）
        if self.inject_sigma <= 0 or not self.training:
            return x
        noise = torch.randn_like(x) * self.inject_sigma
        return x + self.noise_scale * noise


class NoiseTranslator(nn.Module):
    """论文里的 T：把真实噪声图翻译成 Gaussian-noisy image。

    输入可以是 1 通道图像，也可以是 image + lambda condition 两通道。
    输出始终是 1 通道的 translated image，并通过全局残差保护原始信号。
    """

    def __init__(
        self,
        input_channels: int = 1,
        width: int = 32,
        middle_blocks: int = 2,
        inject_sigma: float = 1.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.residual_scale = float(residual_scale)

        self.intro = nn.Conv2d(self.input_channels, width, kernel_size=3, padding=1, padding_mode="reflect")
        self.enc1 = GaussianInjectionBlock(width, inject_sigma=inject_sigma)
        self.down1 = nn.Conv2d(width, width * 2, kernel_size=2, stride=2)
        self.enc2 = GaussianInjectionBlock(width * 2, inject_sigma=inject_sigma)
        self.down2 = nn.Conv2d(width * 2, width * 4, kernel_size=2, stride=2)

        self.middle = nn.Sequential(
            *[GaussianInjectionBlock(width * 4, inject_sigma=inject_sigma) for _ in range(middle_blocks)]
        )

        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, kernel_size=2, stride=2)
        self.dec2 = GaussianInjectionBlock(width * 2, inject_sigma=inject_sigma)
        self.up1 = nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2)
        self.dec1 = GaussianInjectionBlock(width, inject_sigma=inject_sigma)
        self.outro = nn.Conv2d(width, 1, kernel_size=3, padding=1, padding_mode="reflect")

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = x[:, :1]
        feat1 = self.enc1(self.intro(x))
        feat2 = self.enc2(self.down1(feat1))
        mid = self.middle(self.down2(feat2))

        x = self._match_size(self.up2(mid), feat2) + feat2
        x = self.dec2(x)
        x = self._match_size(self.up1(x), feat1) + feat1
        x = self.dec1(x)
        delta = self.outro(x)
        return image + self.residual_scale * delta
