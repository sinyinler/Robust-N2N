# -*- coding: utf-8 -*-
"""跨视图特征一致性（SimSiam 式：projector(可选) + predictor + 归一化对称负余弦 + stop-grad）。

对 out3、bridge：
  use_proj=1: z = φ_s(feat)（单个 1×1 卷积 projector，通道→dim）;  p = h_s(z)
  use_proj=0: z = feat（直接用编码器特征、不加 projector，通道=原特征通道）; p = h_s(z)
  L_s = 0.5·(−cos(p1, sg·z2)) + 0.5·(−cos(p2, sg·z1))          # 归一化后逐像素余弦
  L_feat = Σ_s w_s · L_s

- **predictor + stop-gradient** 是防塌缩必需件（去掉 predictor 会塌，见 SimSiam）。
- **projector（use_proj=1）** 起"解耦缓冲"作用：让 SSL 对齐压在投影空间、不直接扭曲解码器要用的编码器特征。
  use_proj=0 时 SSL 直接压在编码器特征上（可能磨细血管/拖累去噪，需盯血管图）。
- 监控：归一化后特征逐维 std（健康≈1/√(z通道)，塌→0）。projector 时 dim=128→~0.088；
  不加 projector 时 out3(64)→~0.125、bridge(80)→~0.112。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PredHead(nn.Module):
    """predictor h：1×1 bottleneck MLP（防塌缩必需件）。"""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, channels, dim: int = 128, pred_hidden: int = 64, weights=None, use_proj: bool = True):
        super().__init__()
        self.use_proj = bool(use_proj)
        if self.use_proj:
            self.projs = nn.ModuleList([nn.Conv2d(c, dim, kernel_size=1, bias=False) for c in channels])
            pred_dims = [dim for _ in channels]                 # predictor 作用在投影后的 dim 上
        else:
            self.projs = None
            pred_dims = list(channels)                          # predictor 直接作用在编码器特征通道上
        self.preds = nn.ModuleList([PredHead(d, pred_hidden) for d in pred_dims])
        self.weights = list(weights) if weights is not None else [1.0] * len(channels)

    @staticmethod
    def _neg_cos(p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def forward(self, feats1, feats2):
        total = feats1[0].new_zeros(())
        stds = []
        for i, (f1, f2, pred, w) in enumerate(zip(feats1, feats2, self.preds, self.weights)):
            z1 = self.projs[i](f1) if self.use_proj else f1     # 带/不带 projector
            z2 = self.projs[i](f2) if self.use_proj else f2
            p1, p2 = pred(z1), pred(z2)
            l = 0.5 * self._neg_cos(p1, z2.detach()) + 0.5 * self._neg_cos(p2, z1.detach())  # stop-grad
            total = total + w * l
            with torch.no_grad():
                stds.append(float(F.normalize(z1, dim=1).std(dim=(0, 2, 3)).mean()))   # ≈1/√(z通道) 健康
        return total, stds
