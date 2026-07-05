# -*- coding: utf-8 -*-
"""暴露编码器多尺度特征的去噪器（无 GIBlock）。

= 原轻量 U-Net(models/denoiser.py, LRB+ELA)，forward 可选返回编码器深层特征，
供跨视图特征一致性(SimSiam 式)损失使用。本分支不再用 GIBlock。

返回的深层特征（"抽象/语义"层，安全区；浅层高分辨率特征不返回，避免强制不变性磨掉细血管）：
  - out3   : 64ch @H/4
  - bridge : 80ch @H/8
"""
from __future__ import annotations

import torch
from torch import nn

from models.denoiser import Encoder, Bridge, Decoder, Transformer_unit


class DenoiserWithFeats(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(input_channels=input_channels)
        self.bridge = Bridge()
        self.decoder = Decoder()
        self.transformer_unit = Transformer_unit()

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        out1, out2, out3 = self.encoder(x)
        bridge = self.bridge(out3)
        decoder_3 = self.decoder(bridge, out1, out2, out3)
        y = self.transformer_unit(decoder_3)
        if return_feats:
            return y, [out3, bridge]   # 深层：out3(64,@H/4), bridge(80,@H/8)
        return y


# 供损失/训练脚本引用的通道数（与上面返回顺序一致）
FEAT_CHANNELS = [64, 80]


if __name__ == "__main__":
    net = DenoiserWithFeats().eval()
    x = torch.randn(1, 1, 256, 256)
    y, feats = net(x, return_feats=True)
    print("out", tuple(y.shape), "| feats", [tuple(f.shape) for f in feats],
          "| params", f"{sum(p.numel() for p in net.parameters())/1e6:.4f} M")
