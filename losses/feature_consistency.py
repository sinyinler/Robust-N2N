# -*- coding: utf-8 -*-
"""跨视图特征一致性损失（SimSiam 式，多尺度/金字塔）。

用 n1、n2 作两个视图，在编码器深层特征上逼"噪声无关"的表征。防塌缩靠 SimSiam 的
**stop-gradient + predictor**（无需 EMA）；再加上重建损失(N2N)本身也拉住编码器，双保险。

每个尺度：
  projector g : 1×1Conv → BN → ReLU → 1×1Conv → BN     （两层 1×1 MLP，非裸线性）
  predictor h : 1×1Conv → BN → ReLU → 1×1Conv          （bottleneck）
  loss = 0.5·D(h(g(f1)), sg(g(f2))) + 0.5·D(h(g(f2)), sg(g(f1))),  D(p,z) = −cos(p,z)（逐像素）
深层权重给大、浅层给小（deep 安全、shallow 易磨血管）。

塌缩监控：归一化投影特征的逐维 std（健康≈1/√dim，塌缩→0），随 loss 一起返回。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ProjHead(nn.Module):
    """projector：两层 1×1（BN+ReLU），给"释放阀"——不强求原始特征相等、只投影相等。"""
    def __init__(self, c_in: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return self.net(x)


class PredHead(nn.Module):
    """predictor：1×1 bottleneck MLP（SimSiam 防塌缩关键之一）。"""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def _neg_cos(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """逐像素负余弦相似度（沿通道归一化后点积，对 B,H,W 取均值）。"""
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, channels, dim: int = 128, pred_hidden: int = 64, weights=None):
        super().__init__()
        self.projs = nn.ModuleList([ProjHead(c, dim) for c in channels])
        self.preds = nn.ModuleList([PredHead(dim, pred_hidden) for _ in channels])
        self.weights = list(weights) if weights is not None else [1.0] * len(channels)
        self.dim = dim

    def forward(self, feats1, feats2):
        total = feats1[0].new_zeros(())
        stds = []
        for f1, f2, g, h, w in zip(feats1, feats2, self.projs, self.preds, self.weights):
            z1, z2 = g(f1), g(f2)
            p1, p2 = h(z1), h(z2)
            l = 0.5 * _neg_cos(p1, z2.detach()) + 0.5 * _neg_cos(p2, z1.detach())
            total = total + w * l
            with torch.no_grad():
                zn = F.normalize(z1, dim=1)                 # 沿通道归一化
                stds.append(float(zn.std(dim=(0, 2, 3)).mean()))  # 逐维 std 均值，≈1/√dim 健康
        return total, stds
