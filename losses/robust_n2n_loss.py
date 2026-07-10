# -*- coding: utf-8 -*-
"""Robust-N2N 一步法组合损失。

L =  α·Charb(f(n1), n2) + β·Charb(f(n2), n1)      # 对称 N2N（Charbonnier 已并入，不再单列）
   + γ·Charb(f(n1), f(n2))                          # 一致性（DenoiseGAN self-constraint 第三项）
   + λ·RTV(f(n1))                                    # 边缘/结构正则

β=γ=0 且 λ=0.01 时，本损失精确退化为原始 N2N 的 `Charb(f(x_in), x_tgt) + 0.01·RTV`；
配合外部的 w_feat·L_feat，即得「N2N + 特征损失」这一干净消融（特征损失是唯一差异）。
权重为 0 的项跳过计算。
   + [训练过半后] w_white·( L_spatial(w) + β_freq·L_freq(w) ),  w = f(n1) − f(n2)   # 残差白度正则

白度作用对象 w=f(n1)−f(n2)：干净信号 y 相消 → 信号无关、不误伤血管；其中残留的相关结构会被
白度损失惩罚，从而驱动去噪器彻底去除空间相关散斑噪声。白度项延迟开、权重小，避免早期扰乱
及"逼出分歧"退化。复用 losses/ntn_losses.py 的尺度自适应实现（用输入自身 std 构造对照高斯）。
"""
from __future__ import annotations

import torch
from torch import nn

from losses.charbonnier import CharbonnierLoss
from losses.rtv import RTVRegularizer
from losses.ntn_losses import ExplicitNoiseTranslationLoss


class RobustN2NLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        w_white: float = 0.05,
        beta_freq: float = 2e-3,
        w_rtv: float = 0.01,
        charb_eps: float = 1e-3,
        rtv_radius: int = 2,
        rtv_sigma: float = 2.0,
        highpass_ratio: float = 0.0,
        self_target: bool = False,
    ):
        super().__init__()
        # self_target=True: 像素靶换成自重建 f(n1)->n1（仅此一项，恒等靶，去掉一致性项），仅供对照实验
        self.self_target = bool(self_target)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.w_white, self.w_rtv = float(w_white), float(w_rtv)
        self.charb = CharbonnierLoss(eps=charb_eps)
        self.rtv = RTVRegularizer(radius=rtv_radius, sigma=rtv_sigma)
        self.explicit = ExplicitNoiseTranslationLoss(beta=beta_freq, highpass_ratio=highpass_ratio)

    def forward(self, f_n1, n1, f_n2, n2, use_whitening: bool = False):
        """n1/n2：同场景两帧噪声图；f_n1=f(n1), f_n2=f(n2)。use_whitening：是否已过半、开白度项。"""
        zero = torch.zeros((), device=f_n1.device, dtype=f_n1.dtype)
        if self.self_target:                        # 对照实验：自重建恒等靶（仅 f(n1)->n1），去掉一致性项
            rec = self.alpha * self.charb(f_n1, n1)  # 只留 f(n1)->n1，不加 f(n2)->n2
            cons = zero
        else:                                        # N2N 重建；beta=0 即退化为单向（= 原始 N2N）
            rec = self.alpha * self.charb(f_n1, n2)
            if self.beta > 0:                        # 权重为 0 的项直接跳过，省无用前向与零梯度
                rec = rec + self.beta * self.charb(f_n2, n1)
            cons = self.gamma * self.charb(f_n1, f_n2) if self.gamma > 0 else zero
        diff = torch.mean(torch.abs(f_n1 - f_n2))   # 未加权 L1：仅作塌缩诊断量，diff→0 表示输出塌成常数
        rtv = self.w_rtv * self.rtv(f_n1) if self.w_rtv > 0 else zero   # w_rtv=0 时跳过 RTV 前向
        total = rec + cons + rtv

        white = torch.zeros((), device=f_n1.device, dtype=f_n1.dtype)
        spatial = freq = white
        if use_whitening and self.w_white > 0:
            w = f_n1 - f_n2                          # 残差（信号 y 相消）
            white_total, spatial, freq = self.explicit(w)
            white = self.w_white * white_total
            total = total + white

        logs = {
            "total": float(total.detach()),
            "rec": float(rec.detach()),
            "cons": float(cons.detach()),
            "diff": float(diff.detach()),
            "rtv": float(rtv.detach()),
            "white": float(white.detach()) if torch.is_tensor(white) else float(white),
            "spatial": float(spatial.detach()) if torch.is_tensor(spatial) else float(spatial),
            "freq": float(freq.detach()) if torch.is_tensor(freq) else float(freq),
        }
        return total, logs
