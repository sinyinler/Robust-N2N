# -*- coding: utf-8 -*-
"""Bias-Free 改造：把网络变成一阶齐次映射 f(a·x)=a·f(x)（a>0），获得跨噪声等级泛化。

依据：Mohan, Kadkhodaie, Simoncelli, Fernandez-Granda, "Robust and Interpretable Blind
Image Denoising via Bias-Free CNNs", ICLR 2020 (arXiv:1906.05478)。核心：去掉网络里
**所有加性常数**，则 f 局部为 f(x)=W(x)·x（W 只依赖 x 的方向、与幅度无关），滤波强度随
噪声等级自适应，从而在训练噪声范围外也能泛化。

散斑噪声在 raw 域是乘性的，log1p 把它拉成加性（Bias-Free 的前提）。故 log1p/expm1 保留，
只把 log 域网络改造成 bias-free。

本网络活跃路径里的加性/非齐次成分只有三处，本模块一次性消除：
  1. 所有 Conv2d/Conv1d 的 bias  → 去掉
  2. BatchNorm2d（减均值 + β 都是加性）→ 换成 BFBatchNorm2d（不减均值、无 β，只除 std·γ）
  3. ELA（GroupNorm 减均值 + Sigmoid 门控，均非齐次）→ 换成 Identity
ReLU、双线性插值本就齐次；Derf/DyT 为死代码（forward 用的是 self.bn）。
"""
from __future__ import annotations

import torch
from torch import nn


class BFBatchNorm2d(nn.BatchNorm2d):
    """Bias-free BatchNorm：不减均值、无 β；按方差归一化后只乘 γ。
    推理(eval)时用固定 running_var → 等价一个固定对角缩放（无加性项），保持一阶齐次。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        yt = x.transpose(0, 1)                                  # (C, N, H, W)
        shp = yt.shape
        y = yt.contiguous().view(x.size(1), -1)                # (C, N*H*W)
        if self.training or not self.track_running_stats:
            var = y.var(dim=1, unbiased=False)                 # 只用方差，不减均值
            if self.training and self.track_running_stats:
                n = y.size(1)
                with torch.no_grad():
                    self.running_var.mul_(1 - self.momentum).add_(
                        self.momentum * var.detach() * n / max(n - 1, 1))
                    self.num_batches_tracked += 1
        else:
            var = self.running_var
        y = y / torch.sqrt(var + self.eps).view(-1, 1)         # 无均值减法
        y = self.weight.view(-1, 1) * y                        # 只乘 γ（乘性，不破坏齐次）；无 β
        return y.view(shp).transpose(0, 1)


def make_bias_free(model: nn.Module) -> nn.Module:
    """就地把 model 改造成 bias-free（去 conv bias / BN→BFBatchNorm / ELA→Identity）。"""
    from models.denoiser import ELA

    # 1) 所有卷积去 bias（1×1 输入/输出 conv 原本带 bias）
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d)) and m.bias is not None:
            m.register_parameter("bias", None)

    # 2) 递归替换 BatchNorm2d → BFBatchNorm2d；ELA → Identity
    def convert(parent: nn.Module):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm2d) and not isinstance(child, BFBatchNorm2d):
                bf = BFBatchNorm2d(child.num_features, eps=child.eps, momentum=child.momentum,
                                   affine=True, track_running_stats=child.track_running_stats)
                bf.weight.data.copy_(child.weight.data)        # 继承 γ
                bf.running_var.data.copy_(child.running_var.data)
                bf.register_parameter("bias", None)            # 显式无 β
                setattr(parent, name, bf)
            elif isinstance(child, ELA):
                setattr(parent, name, nn.Identity())           # 去掉 sigmoid 门控
            else:
                convert(child)

    convert(model)
    return model


def count_additive_constants(model: nn.Module) -> dict:
    """自检：改造后残留的加性/非齐次成分数（应全为 0）。"""
    from models.denoiser import ELA
    n_bias = sum(1 for m in model.modules()
                 if isinstance(m, (nn.Conv2d, nn.Conv1d)) and m.bias is not None)
    n_bn = sum(1 for m in model.modules()
               if isinstance(m, nn.BatchNorm2d) and not isinstance(m, BFBatchNorm2d))
    n_ela = sum(1 for m in model.modules() if isinstance(m, ELA))
    n_bf_bias = sum(1 for m in model.modules()
                    if isinstance(m, BFBatchNorm2d) and getattr(m, "bias", None) is not None)
    return {"conv_bias": n_bias, "std_batchnorm": n_bn, "ela": n_ela, "bf_beta": n_bf_bias}


if __name__ == "__main__":
    from models.denoiser_feats import DenoiserWithFeats
    net = DenoiserWithFeats(input_channels=1)
    print("改造前:", count_additive_constants(net))
    make_bias_free(net)
    print("改造后:", count_additive_constants(net), "（应全 0）")
    net.eval()
    x = torch.randn(2, 1, 128, 128).abs()
    with torch.no_grad():
        y1 = net(x)
        y2 = net(3.0 * x)
    rel = (y2 - 3.0 * y1).norm() / (3.0 * y1).norm().clamp_min(1e-8)
    print(f"齐次性自检 ‖f(3x)−3f(x)‖/‖3f(x)‖ = {rel.item():.2e} （应≈0）")
