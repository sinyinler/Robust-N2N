# -*- coding: utf-8 -*-
"""跨视图特征一致性（选项 A：1×1 卷积提特征 + L2 归一化 + 对称负余弦 + stop-grad）。

对 out3、bridge 各用一个 1×1 卷积 φ 提特征，**沿通道 L2 归一化**后算对称负余弦，
目标侧 stop-grad：
    z = φ_s(feat) ;  p = normalize(z, dim=channel)
    L_s = 0.5·(−cos(p1, sg·p2)) + 0.5·(−cos(p2, sg·p1))          # 逐像素余弦，对 B,H,W 取均值
    L_feat = Σ_s  w_s · L_s

要点：
- **归一化 → 尺度无关**，L_s ∈ [−1,1]，w_feat=0.1 就能实打实起作用（解决 v6 特征损失量级 ~1e-3 太小的问题）；
- **stop-grad → 防塌缩**（重建损失再兜一层）；单个 1×1 卷积，不带 predictor/MLP（不够再加）。
- 塌缩监控：归一化特征逐维 std（健康≈1/√dim，塌→0）。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, channels, dim: int = 128, weights=None):
        super().__init__()
        # 每个尺度一个 1×1 卷积（逐像素线性投影）
        self.projs = nn.ModuleList([nn.Conv2d(c, dim, kernel_size=1, bias=False) for c in channels])
        self.weights = list(weights) if weights is not None else [1.0] * len(channels)
        self.dim = dim

    @staticmethod
    def _neg_cos(p, z):
        return -(p * z).sum(dim=1).mean()   # p,z 已沿通道归一化 → 逐像素余弦；对 B,H,W 取均值

    def forward(self, feats1, feats2):
        total = feats1[0].new_zeros(())
        stds = []
        for f1, f2, proj, w in zip(feats1, feats2, self.projs, self.weights):
            p1 = F.normalize(proj(f1), dim=1)      # L2 归一化（沿通道）
            p2 = F.normalize(proj(f2), dim=1)
            l = 0.5 * self._neg_cos(p1, p2.detach()) + 0.5 * self._neg_cos(p2, p1.detach())  # 对称 + stop-grad
            total = total + w * l
            with torch.no_grad():
                stds.append(float(p1.std(dim=(0, 2, 3)).mean()))   # 逐维 std，≈1/√dim 健康
        return total, stds
