import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_2d(ksize: int, sigma: float, device, dtype):
    # ksize 必须是奇数
    r = (ksize - 1) / 2.0
    ax = torch.arange(ksize, device=device, dtype=dtype) - r
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k = k / (k.sum() + 1e-12)
    return k  # [K,K]


class RTVRegularizer(nn.Module):
    """
    RTV 正则：更倾向“保大结构边缘、压小尺度纹理/噪声边缘”
    用法：loss_rtv = rtv(pred)  -> 标量

    参数：
      radius: 窗口半径，窗口大小 = 2*radius+1（越大越只保大尺度边缘）
      sigma : 窗口高斯权重的 sigma
      eps   : 防止除 0
    """
    def __init__(self, radius: int = 2, sigma: float = 2.0, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.radius = int(radius)
        self.sigma = float(sigma)
        self.eps = float(eps)
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

        ksize = 2 * self.radius + 1
        # 先存 CPU kernel，forward 时再转到对应 device/dtype
        kernel = _gaussian_kernel_2d(ksize, self.sigma, device="cpu", dtype=torch.float32)
        self.register_buffer("_kernel_cpu", kernel)

    @staticmethod
    def _dx(x):
        # 前向差分，保持同尺寸
        d = x[..., :, 1:] - x[..., :, :-1]
        return F.pad(d, (0, 1, 0, 0), mode="replicate")

    @staticmethod
    def _dy(x):
        d = x[..., 1:, :] - x[..., :-1, :]
        return F.pad(d, (0, 0, 0, 1), mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B, C, H, W = x.shape

        dx = self._dx(x)
        dy = self._dy(x)

        # depthwise conv 的权重：每个 channel 共用同一个高斯核
        k = self._kernel_cpu.to(device=x.device, dtype=x.dtype)  # [K,K]
        k = k.unsqueeze(0).unsqueeze(0)  # [1,1,K,K]
        weight = k.expand(C, 1, k.shape[-2], k.shape[-1]).contiguous()  # [C,1,K,K]
        pad = self.radius

        # WTV：窗口内 |grad| 的加权和
        WTVx = F.conv2d(dx.abs(), weight, padding=pad, groups=C)
        WTVy = F.conv2d(dy.abs(), weight, padding=pad, groups=C)

        # WIV：窗口内 grad(带符号) 的加权和再 abs（纹理会抵消，结构不抵消）
        WIVx = F.conv2d(dx, weight, padding=pad, groups=C).abs()
        WIVy = F.conv2d(dy, weight, padding=pad, groups=C).abs()

        rtv_map = WTVx / (WIVx + self.eps) + WTVy / (WIVy + self.eps)

        if self.reduction == "mean":
            return rtv_map.mean()
        if self.reduction == "sum":
            return rtv_map.sum()
        return rtv_map
