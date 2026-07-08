# -*- coding: utf-8 -*-
"""跨视图特征一致性（SimSiam 式：projector(可选) + predictor + 归一化对称负余弦 + stop-grad）。

对 out3、bridge：
  use_proj=1: z = g_s(feat)（SimSiam 论文式 3 层 MLP projector，每层 BN、末层无 ReLU）;  p = h_s(z)
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


class ProjHead(nn.Module):
    """projector g：SimSiam 式 3 层 MLP（每层带 BN；隐藏层有 ReLU，末层有 BN 无 ReLU）。1×1 卷积实现。"""
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim),          # 末层：有 BN、无 ReLU
        )

    def forward(self, x):
        return self.net(x)


class PredHead(nn.Module):
    """predictor h：SimSiam 式 2 层 bottleneck MLP（隐藏层 BN+ReLU，输出层无 BN 无 ReLU）。防塌缩必需件。"""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, channels, dim: int = 128, pred_hidden=None, weights=None, use_proj: bool = True):
        super().__init__()
        self.use_proj = bool(use_proj)
        # dim<=0（或 None）: 每个尺度用原生通道 C（projector C→C→C）
        native = self.use_proj and ((dim is None) or (int(dim) <= 0))
        if self.use_proj:
            self.proj_dims = [c if native else int(dim) for c in channels]      # 各尺度投影维度
            self.projs = nn.ModuleList([ProjHead(c, d) for c, d in zip(channels, self.proj_dims)])
        else:
            self.projs = None
            self.proj_dims = list(channels)                     # 无 projector：z=原生特征
        # predictor 瓶颈：不显式指定则自动取 dim//4（SimSiam 论文推荐比例；=dim 会不稳/失败）
        auto_h = (pred_hidden is None) or (int(pred_hidden) <= 0)
        self.pred_hiddens = [max(4, d // 4) if auto_h else int(pred_hidden) for d in self.proj_dims]
        self.preds = nn.ModuleList([PredHead(d, h) for d, h in zip(self.proj_dims, self.pred_hiddens)])
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
