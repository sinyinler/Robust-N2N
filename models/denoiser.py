import torch
from torch import nn
import torch.nn.functional as F


# ==========================================
# 1. 新增 DyT 模块
# ==========================================
class DyT(nn.Module):
    """
    Dynamic Tanh (DyT) layer adapted for CNNs (NCHW format).
    Formula: DyT(x) = gamma * tanh(alpha * x) + beta
    """

    def __init__(self, channels, init_alpha=0.5):
        super(DyT, self).__init__()
        # alpha 是一个可学习的标量，用于控制 tanh 的输入范围 [cite: 209]
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

        # gamma 和 beta 是通道维度的缩放和平移参数，类似于 BN/LN 的 affine parameters [cite: 214]
        # 形状设为 (1, C, 1, 1) 以便在 NCHW 格式下进行广播
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # 逐元素操作
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class Derf(nn.Module):
    """
    Dynamic erf (Derf) layer.
    Paper: Stronger Normalization-Free Transformers (arXiv:2512.10938)
    Formula: Derf(x) = gamma * erf(alpha * x + s) + beta

    Args:
        channels (int): Number of input channels.
        init_alpha (float): Initial value for alpha. Default: 0.5 (per paper).
        init_s (float): Initial value for s. Default: 0.0 (per paper).
    """

    def __init__(self, channels, init_alpha=0.5, init_s=0.0):
        super(Derf, self).__init__()
        # alpha 和 s 是可学习的标量 (scalar)
        # 论文 Section 7.1 提到 scalar s 和 vector s 效果差别不大，为了效率使用 scalar
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.s = nn.Parameter(torch.tensor(init_s))

        # gamma 和 beta 是通道维度的参数 (per-channel vectors)
        # 形状设为 (1, C, 1, 1) 以便在 CNN (NCHW) 中广播
        # 如果是 Transformer (N, L, C)，代码会自动广播，或者需要调整形状为 (1, 1, C)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # x: (N, C, H, W) for CNN
        # 核心公式: erf(alpha * x + s)
        # torch.erf 是高斯误差函数
        scaled_input = torch.erf(self.alpha * x + self.s)

        # 应用仿射变换 (affine transformation)
        return self.gamma * scaled_input + self.beta
    
#增加ELA模块
class ELA(nn.Module):
    def __init__(self, channel, kernel_size: int = 7, groups: int = 8):
        super(ELA, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须是奇数(如 3/5/7)"
        assert channel % groups == 0, f"GroupNorm要求 channel({channel}) 能被 groups({groups}) 整除"

        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(
            channel, channel,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=channel,      # depthwise 1D conv
            bias=False
        )
        self.gn = nn.GroupNorm(groups, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Height attention: 对宽度方向求均值，得到 (B, C, H)
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # Width attention: 对高度方向求均值，得到 (B, C, W)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        return x * x_h * x_w


# ==========================================
# 2. 原有模块（无需修改）
# ==========================================
class CONv1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CONv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x


# ==========================================
# 3. 修改 Light_Residual_block (BN -> DyT)
# ==========================================
class Light_Residual_block(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int = 1,
                 use_ela: bool = False,
                 ela_kernel_size: int = 7,
                 ela_groups: int = 8):
        super(Light_Residual_block, self).__init__()

        self.DWConv_1 = nn.Conv2d(
            in_channels=input_channels,
            groups=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            padding_mode='reflect',
            dilation=dilation,
            bias=False
        )

        self.PWConv_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # 你原来的归一化（BN / DyT / Derf 三选一）
        self.bn = nn.BatchNorm2d(output_channels)
        # self.dyt = DyT(output_channels)
        # self.norm = Derf(output_channels)

        self.relu = nn.ReLU()

        self.DWConv_2 = nn.Conv2d(
            in_channels=output_channels,
            groups=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=dilation * (kernel_size // 2),
            padding_mode='reflect',
            dilation=dilation,
            bias=False
        )

        self.PWConv_2 = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # ===== 新增：ELA（只作用在残差分支）=====
        if use_ela:
            self.ela = ELA(channel=output_channels,
                           kernel_size=ela_kernel_size,
                           groups=ela_groups)
        else:
            self.ela = nn.Identity()
        # =======================================

        if (input_channels != output_channels) or (stride != 1):
            self.short_cut = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            )
        else:
            self.short_cut = nn.Identity()

    def forward(self, x):
        out = self.DWConv_1(x)
        out = self.PWConv_1(out)

        # 归一化层（你根据实验选）
        out = self.bn(out)
        # out = self.dyt(out)
        # out = self.norm(out)

        out = self.relu(out)
        out = self.DWConv_2(out)
        out = self.PWConv_2(out)

        # ===== 新增：ELA 放在 PWConv_2 后、残差相加前 =====
        out = self.ela(out)
        # ===============================================

        out = out + self.short_cut(x)
        out = self.relu(out)
        return out


# ==========================================
# 4. 后续网络结构保持不变
# ==========================================

class Transformer_unit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, decoder_1):
        tn1 = self.conv1_1(decoder_1)
        return tn1


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.Conv1x1 = CONv1x1(in_channel=input_channels, out_channel=1)
        self.Light_Residual_block_1 = Light_Residual_block(input_channels=1, output_channels=16, kernel_size=3,
                                                           stride=1, dilation=1,use_ela=False)
        self.Light_Residual_block_2 = Light_Residual_block(input_channels=16, output_channels=32, kernel_size=3,
                                                           stride=2, dilation=1,use_ela=False)
        self.Light_Residual_block_3 = Light_Residual_block(input_channels=32, output_channels=64, kernel_size=3,
                                                           stride=2, dilation=1,use_ela=True,ela_kernel_size=7,ela_groups=8)

    def forward(self, x):
        x=self.Conv1x1(x)
        out_1 = self.Light_Residual_block_1(x)
        out_2 = self.Light_Residual_block_2(out_1)
        out_3 = self.Light_Residual_block_3(out_2)
        return out_1, out_2, out_3


class Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.bridge = Light_Residual_block(input_channels=64, output_channels=80, kernel_size=3, stride=2, dilation=1,use_ela=True,ela_kernel_size=7,ela_groups=8)

    def forward(self, out_3):
        out = self.bridge(out_3)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_1 = Light_Residual_block(input_channels=144, output_channels=64, kernel_size=3, stride=1,
                                              dilation=1,use_ela=True,ela_kernel_size=7,ela_groups=8)
        self.decoder_2 = Light_Residual_block(input_channels=96, output_channels=32, kernel_size=3, stride=1,
                                              dilation=1)
        self.decoder_3 = Light_Residual_block(input_channels=48, output_channels=16, kernel_size=3, stride=1,
                                              dilation=1)

    def forward(self, out, out_1, out_2, out_3):
        up_1 = F.interpolate(out, size=out_3.shape[-2:], mode='bilinear', align_corners=False)
        cat1 = torch.cat((up_1, out_3), dim=1)
        decoder_1 = self.decoder_1(cat1)

        up_2 = F.interpolate(decoder_1, size=out_2.shape[-2:], mode='bilinear', align_corners=False)
        cat2 = torch.cat((up_2, out_2), dim=1)
        decoder_2 = self.decoder_2(cat2)

        up_3 = F.interpolate(decoder_2, size=out_1.shape[-2:], mode='bilinear', align_corners=False)
        cat3 = torch.cat((up_3, out_1), dim=1)
        decoder_3 = self.decoder_3(cat3)

        return decoder_3


class Denoiser(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(input_channels=input_channels)
        self.bridge = Bridge()
        self.decoder = Decoder()
        self.transformer_unit = Transformer_unit()

    def forward(self, x):
        out1, out2, out3 = self.encoder(x)
        bridge = self.bridge(out3)
        decoder_3 = self.decoder(bridge, out1, out2, out3)
        transformed_decoder_1 = self.transformer_unit(decoder_3)
        return transformed_decoder_1


if __name__ == "__main__":
    net = Denoiser().cpu()
    x = torch.randn(1, 1, 680, 680)  # 保持你的输入尺寸
    t1 = net(x)
    print("---------------------------------------")
    print(f"Model Output Shapes: {t1.shape}")
    print("---------------------------------------")
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total Params: {total_params / 1e6:.4f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.4f} M")
