# -*- coding: utf-8 -*-
"""Robust-N2N 去噪器：在原轻量 U-Net(models/denoiser.py) 的三个位置插入 GIBlock。

插入位置（与用户确认）：
  1. 最深编码块 LRB3 的输出 (64ch, @H/4)
  2. 瓶颈 Bridge 的输出 (80ch, @H/8)
  3. 第一个解码块 decoder_1 的输出 (64ch, @H/4)

GIBlock = NAFBlock + 可学习强度的 Gaussian 注入；注入只在训练时启用、推理时关闭
（见 models/ntn.py 的 GaussianInjectionBlock，已按 self.training gating）。
其余骨干（Encoder/Bridge/Decoder/Transformer_unit）完全复用原去噪器，保持轻量。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.denoiser import Encoder, Bridge, Decoder, Transformer_unit
from models.ntn import GaussianInjectionBlock


class RobustDenoiser(nn.Module):
    def __init__(self, input_channels: int = 1, inject_sigma: float = 1.0, init_noise_scale: float = 0.1):
        super().__init__()
        self.encoder = Encoder(input_channels=input_channels)
        self.bridge = Bridge()
        self.decoder = Decoder()
        self.transformer_unit = Transformer_unit()
        # 三处 GIBlock（通道数与对应特征一致）
        self.gib_enc = GaussianInjectionBlock(64, inject_sigma=inject_sigma, init_noise_scale=init_noise_scale)
        self.gib_mid = GaussianInjectionBlock(80, inject_sigma=inject_sigma, init_noise_scale=init_noise_scale)
        self.gib_dec = GaussianInjectionBlock(64, inject_sigma=inject_sigma, init_noise_scale=init_noise_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, out2, out3 = self.encoder(x)
        out3 = self.gib_enc(out3)                 # ① 最深编码块后注入
        bridge = self.bridge(out3)
        bridge = self.gib_mid(bridge)             # ② 瓶颈后注入

        # 展开 Decoder.forward，以便在 decoder_1 之后插入 GIBlock
        up_1 = F.interpolate(bridge, size=out3.shape[-2:], mode="bilinear", align_corners=False)
        decoder_1 = self.decoder.decoder_1(torch.cat((up_1, out3), dim=1))
        decoder_1 = self.gib_dec(decoder_1)       # ③ 第一个解码块后注入

        up_2 = F.interpolate(decoder_1, size=out2.shape[-2:], mode="bilinear", align_corners=False)
        decoder_2 = self.decoder.decoder_2(torch.cat((up_2, out2), dim=1))

        up_3 = F.interpolate(decoder_2, size=out1.shape[-2:], mode="bilinear", align_corners=False)
        decoder_3 = self.decoder.decoder_3(torch.cat((up_3, out1), dim=1))

        return self.transformer_unit(decoder_3)


if __name__ == "__main__":
    net = RobustDenoiser().eval()
    x = torch.randn(1, 1, 256, 256)
    y1 = net(x)
    # 推理时 GIBlock 关闭 → 两次前向应完全一致（确定性）
    y2 = net(x)
    net.train()
    yt = net(x)  # 训练时注入生效
    tot = sum(p.numel() for p in net.parameters())
    print("out", tuple(y1.shape), "| eval deterministic:", torch.allclose(y1, y2),
          "| train injects noise:", not torch.allclose(y1, yt))
    print(f"params: {tot/1e6:.4f} M")
