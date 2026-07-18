"""SIDD sRGB 三通道监督去噪网络。

该模块刻意不修改原有单通道 ``models.denoiser.Denoiser``，避免 SIDD 实验
改变既有 BFI/N2N checkpoint 的结构和行为。主体仍复用相同的轻量残差块，
但 RGB 输入不会先压成灰度，输出也固定为三个颜色通道。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.denoiser import Light_Residual_block


class SIDDRGBDenoiser(nn.Module):
    """轻量 RGB U-Net，直接预测 [0, 1] sRGB 域中的干净图像。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = Light_Residual_block(3, 16, 3, 1, use_ela=False)
        self.encoder2 = Light_Residual_block(16, 32, 3, 2, use_ela=False)
        self.encoder3 = Light_Residual_block(
            32, 64, 3, 2, use_ela=True, ela_kernel_size=7, ela_groups=8
        )
        self.bridge = Light_Residual_block(
            64, 80, 3, 2, use_ela=True, ela_kernel_size=7, ela_groups=8
        )
        self.decoder1 = Light_Residual_block(
            144, 64, 3, 1, use_ela=True, ela_kernel_size=7, ela_groups=8
        )
        self.decoder2 = Light_Residual_block(96, 32, 3, 1)
        self.decoder3 = Light_Residual_block(48, 16, 3, 1)
        self.output = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.encoder1(x)
        out2 = self.encoder2(out1)
        out3 = self.encoder3(out2)
        bridge = self.bridge(out3)

        up1 = F.interpolate(bridge, size=out3.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.decoder1(torch.cat((up1, out3), dim=1))
        up2 = F.interpolate(dec1, size=out2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.decoder2(torch.cat((up2, out2), dim=1))
        up3 = F.interpolate(dec2, size=out1.shape[-2:], mode="bilinear", align_corners=False)
        dec3 = self.decoder3(torch.cat((up3, out1), dim=1))
        return self.output(dec3)


if __name__ == "__main__":
    model = SIDDRGBDenoiser()
    sample = torch.rand(2, 3, 256, 256)
    output = model(sample)
    print("output", tuple(output.shape))
    print("params", sum(p.numel() for p in model.parameters()))
