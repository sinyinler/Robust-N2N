# -*- coding: utf-8 -*-
"""跨视图特征一致性（1×1 卷积提取特征 + Charbonnier 差）。

对 out3、bridge 各用一个 1×1 卷积 φ 把特征投影，再对两视图的投影特征算 Charbonnier 差：
    L_feat = Σ_s  w_s · Charbonnier( φ_s(feat_n1) , φ_s(feat_n2) )

注意：本版**去掉了 SimSiam 的 projector/predictor 与 stop-gradient**（按用户要求改为裸 1×1 卷积
+ 直接拉近特征）。因此**有表征塌缩风险**（编码器/投影可能退化成常数特征使差=0）。靠：
  ① 重建损失(对称 N2N)兜住输出；② std 监控（归一化投影特征逐维标准差，健康≈1/√dim，塌→0）。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, channels, dim: int = 128, charb_eps: float = 1e-3, weights=None):
        super().__init__()
        # 每个尺度一个 1×1 卷积（逐像素线性投影），out3/bridge 通道不同故各一。
        self.projs = nn.ModuleList([nn.Conv2d(c, dim, kernel_size=1, bias=False) for c in channels])
        self.weights = list(weights) if weights is not None else [1.0] * len(channels)
        self.eps = float(charb_eps)
        self.dim = dim

    def forward(self, feats1, feats2):
        total = feats1[0].new_zeros(())
        stds = []
        for f1, f2, proj, w in zip(feats1, feats2, self.projs, self.weights):
            z1, z2 = proj(f1), proj(f2)                       # 1×1 卷积提取特征
            d = z1 - z2
            l = torch.mean(torch.sqrt(d * d + self.eps * self.eps))   # Charbonnier(z1, z2)
            total = total + w * l
            with torch.no_grad():
                zn = F.normalize(z1, dim=1)                   # 沿通道归一化
                stds.append(float(zn.std(dim=(0, 2, 3)).mean()))  # 逐维 std 均值，≈1/√dim 健康
        return total, stds
