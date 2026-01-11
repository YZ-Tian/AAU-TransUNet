import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveUpsampling(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveUpsampling, self).__init__()

        # 转置卷积用于细节恢复
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )

        # 权重预测模块（生成空间权重图）
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 转置卷积上采样
        transconv_out = self.transposed_conv(x)

        # 双线性插值上采样（全局结构保持）
        interp_out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # 生成权重图并调整尺寸
        weight = self.weight_generator(x)
        weight = F.interpolate(weight, size=transconv_out.size()[2:], mode='nearest')

        # 加权融合
        output = weight * transconv_out + (1 - weight) * interp_out

        return output