import torch
import torch.nn as nn

class TVLoss(nn.Module):
    """
    Total Variation Loss (L1 Regularization on gradients).
    用于约束图像的平滑度，去除噪点。
    """
    def __init__(self, reduction: str = 'mean'):
        super(TVLoss, self).__init__()
        assert reduction in ['mean', 'sum'], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor = None):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 (在此 Loss 中不使用，仅为了接口兼容)
        """
        # 计算水平方向梯度 (B, C, H, W-1)
        w_diff = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        # 计算垂直方向梯度 (B, C, H-1, W)
        h_diff = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        # 直接分别计算 L1 范数并求和
        if self.reduction == 'sum':
            loss = torch.abs(w_diff).sum() + torch.abs(h_diff).sum()
        else: # mean
            # 对两个方向的梯度分别求均值然后相加
            loss = torch.abs(w_diff).mean() + torch.abs(h_diff).mean()

        return loss