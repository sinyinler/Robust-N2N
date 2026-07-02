import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction="mean"):
        """
        Args:
            delta: Huber 的拐点（很关键！按你 Box-Cox 域数据尺度来调）
            reduction: "mean" / "sum" / "none"
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.loss_fn = nn.HuberLoss(delta=delta, reduction=reduction)

    def forward(self, x, y):
        return self.loss_fn(x, y)
