# -*- coding: utf-8 -*-
"""跨视图特征一致性（选项 A + predictor：单 1×1 卷积提特征 + predictor + 归一化 + 对称负余弦 + stop-grad）。

对 out3、bridge：单个 1×1 卷积 φ 提特征 → predictor h → L2 归一化 → 对称负余弦、目标侧 stop-grad：
    z = φ_s(feat) ;  p = h(z)
    L_s = 0.5·(−cos(p1, sg·z2)) + 0.5·(−cos(p2, sg·z1))          # 逐像素余弦，归一化后算
    L_feat = Σ_s  w_s · L_s

**predictor + stop-gradient 是 SimSiam 防塌缩的必需件**（只有 stop-grad、没有 predictor 会塌，
见 SimSiam 消融；v7 无 predictor 立即塌成 std≈0）。这里保留"单个 1×1 卷积提特征"，只补回 predictor。
监控：归一化投影特征逐维 std（健康≈1/√dim，塌→0）。
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
    def __init__(self, channels, dim: int = 128, pred_hidden: int = 64, weights=None):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, dim, kernel_size=1, bias=False) for c in channels])  # 单 1×1 卷积
        self.preds = nn.ModuleList([PredHead(dim, pred_hidden) for _ in channels])                    # predictor
        self.weights = list(weights) if weights is not None else [1.0] * len(channels)
        self.dim = dim

    @staticmethod
    def _neg_cos(p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def forward(self, feats1, feats2):
        total = feats1[0].new_zeros(())
        stds = []
        for f1, f2, proj, pred, w in zip(feats1, feats2, self.projs, self.preds, self.weights):
            z1, z2 = proj(f1), proj(f2)
            p1, p2 = pred(z1), pred(z2)
            # 预测侧过 predictor，目标侧 stop-grad（z 不回传）→ 防塌缩
            l = 0.5 * self._neg_cos(p1, z2.detach()) + 0.5 * self._neg_cos(p2, z1.detach())
            total = total + w * l
            with torch.no_grad():
                zn = F.normalize(z1, dim=1)
                stds.append(float(zn.std(dim=(0, 2, 3)).mean()))   # ≈1/√dim 健康
        return total, stds
