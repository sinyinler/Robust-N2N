# -*- coding: utf-8 -*-
"""无 skip 的辅助解码器：只吃 bridge，重建整幅图。

**动机（由诊断实测得出）**：在当前 U-Net 里，场景特异信息约 90% 走浅层 skip，
bottleneck 只承载约 10%（batch_shuffle 占天花板 11.6%），且其有效秩仅 30/80。
把「纯不变性」目标（SimSiam 一致性）挂在这样一条低带宽支路上，最省力的满足方式
就是**继续丢信息**——实测有效秩被压到 8.4/80，份额降到 8.7%。

因此这里加的不是「移除变化」的目标，而是「**要求信息存在**」的目标：
    n1 → encoder → bridge → AuxDecoder（无 skip）→ ŷ_aux
    L_aux = Charbonnier(ŷ_aux, n2)          # 靶子仍是独立噪声的兄弟帧 → N2N 式，无需干净 GT

没有 skip，bridge 就**必须**携带足以重建整幅图的场景结构，从架构上禁止「降秩省事」。
训练专用；不属于 Denoiser，不进 checkpoint，推理时不存在。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.denoiser import Light_Residual_block


class AuxDecoder(nn.Module):
    def __init__(self, in_channels: int = 80):
        super().__init__()
        # bridge 在 H/8，逐级上采样回全分辨率；通道数镜像主解码器
        self.block1 = Light_Residual_block(input_channels=in_channels, output_channels=64,
                                           kernel_size=3, stride=1, dilation=1)
        self.block2 = Light_Residual_block(input_channels=64, output_channels=32,
                                           kernel_size=3, stride=1, dilation=1)
        self.block3 = Light_Residual_block(input_channels=32, output_channels=16,
                                           kernel_size=3, stride=1, dilation=1)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, bridge: torch.Tensor, out_size) -> torch.Tensor:
        """bridge: (N,80,H/8,W/8) → (N,1,H,W)。out_size 为目标 (H,W)，兼容奇数尺寸。"""
        x = self.block1(bridge)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block3(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.out(x)


if __name__ == "__main__":
    net = AuxDecoder(80).eval()
    y = net(torch.randn(2, 80, 64, 64), (512, 512))
    print("out", tuple(y.shape), "| params", f"{sum(p.numel() for p in net.parameters()) / 1e6:.4f} M")
