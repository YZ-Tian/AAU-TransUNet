import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, C_enc, C_dec, C_mid=16, use_bn=True, use_residual=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_residual = use_residual
        
        # 特征对齐的1x1卷积
        self.proj_enc = nn.Conv2d(C_enc, C_mid, kernel_size=1, stride=1, padding=0, bias=not use_bn)
        self.proj_dec = nn.Conv2d(C_dec, C_mid, kernel_size=1, stride=1, padding=0, bias=not use_bn)
        
        # 批归一化层
        if use_bn:
            self.bn_enc = nn.BatchNorm2d(C_mid)
            self.bn_dec = nn.BatchNorm2d(C_mid)
            self.bn_att = nn.BatchNorm2d(C_mid)
        
        # 注意力掩码生成
        self.attention_conv = nn.Conv2d(C_mid, 1, kernel_size=1, stride=1, padding=0)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_enc, x_dec):
        """
        Args:
            x_enc: 编码器特征 [B, C_enc, H, W]
            x_dec: 解码器特征 [B, C_dec, H, W]
        Returns:
            x_enc_attended: 注意力加权的编码器特征 [B, C_enc, H, W]
        """
        # 特征对齐和归一化
        x_enc_proj = self.proj_enc(x_enc)
        x_dec_proj = self.proj_dec(x_dec)
        
        if self.use_bn:
            x_enc_proj = self.bn_enc(x_enc_proj)
            x_dec_proj = self.bn_dec(x_dec_proj)
        
        # 计算注意力掩码
        attention = self.relu(x_enc_proj + x_dec_proj)
        if self.use_bn:
            attention = self.bn_att(attention)
        
        attention_mask = self.sigmoid(self.attention_conv(attention))  # [B, 1, H, W]
        
        # 编码器特征加权
        x_enc_attended = x_enc * attention_mask
        
        # 残差连接
        if self.use_residual:
            x_enc_attended = x_enc_attended + x_enc
        
        return x_enc_attended


class MultiScaleAttentionGate(nn.Module):
    """多尺度注意力门控，适用于不同分辨率的特征图"""
    def __init__(self, C_enc, C_dec, C_mid=16, use_bn=True, use_residual=True):
        super().__init__()
        self.attention_gate = AttentionGate(C_enc, C_dec, C_mid, use_bn, use_residual)
        
    def forward(self, x_enc, x_dec):
        # 确保特征图尺寸匹配
        if x_enc.shape[2:] != x_dec.shape[2:]:
            x_dec = F.interpolate(x_dec, size=x_enc.shape[2:], mode='bilinear', align_corners=False)
        
        return self.attention_gate(x_enc, x_dec)


# 使用示例
if __name__ == "__main__":
    # 测试 AttentionGate
    batch_size, channels_enc, channels_dec, height, width = 2, 64, 128, 32, 32
    
    ag = AttentionGate(C_enc=channels_enc, C_dec=channels_dec, C_mid=32)
    
    x_enc = torch.randn(batch_size, channels_enc, height, width)
    x_dec = torch.randn(batch_size, channels_dec, height, width)
    
    output = ag(x_enc, x_dec)
    print(f"Input encoder shape: {x_enc.shape}")
    print(f"Input decoder shape: {x_dec.shape}")
    print(f"Output shape: {output.shape}")
    print(f"AttentionGate parameters: {sum(p.numel() for p in ag.parameters()):,}")