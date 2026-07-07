# -*- coding: utf-8 -*-
"""跨视图特征一致性（最小改动版：1×1 卷积提特征 → L2 归一化 → Charbonnier 差）。

对 out3、bridge 各用一个 1×1 卷积 φ 提特征，**沿通道 L2 归一化（尺度无关）**后算 Charbonnier：
    z = normalize( φ_s(feat) , dim=channel )
    L_feat = Σ_s  w_s · Charbonnier( z1 , z2 )        # z1,z2 为两视图归一化后的投影特征

归一化解决 v6"Charbonnier 特征距离量级 ~1e-3、太小不起作用"的问题（归一化后量级 O(1)，w_feat 生效）。
不含 predictor / stop-grad（按用户"一点点改"要求，先只加归一化）。
⚠ 提示：归一化 + 直接拉近特征，平凡解仍是"常向量"，**可能塌**（盯 std：健康≈1/√dim，塌→0）。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, channels, dim: int = 128, charb_eps: float = 1e-3, weights=None):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, dim, kernel_size=1, bias=False) for c in channels])
        self.weights = list(weights) if weights is not None else [1.0] * len(channels)
        self.eps = float(charb_eps)
        self.dim = dim

    def forward(self, feats1, feats2):
        total = feats1[0].new_zeros(())
        stds = []
        for f1, f2, proj, w in zip(feats1, feats2, self.projs, self.weights):
            z1 = F.normalize(proj(f1), dim=1)      # 1×1 卷积 + L2 归一化（尺度无关）
            z2 = F.normalize(proj(f2), dim=1)
            d = z1 - z2
            l = torch.mean(torch.sqrt(d * d + self.eps * self.eps))   # Charbonnier(z1, z2)
            total = total + w * l
            with torch.no_grad():
                stds.append(float(z1.std(dim=(0, 2, 3)).mean()))       # ≈1/√dim 健康
        return total, stds
