import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from AttentionGate import AttentionGate
from Adaptive_Upsampling_Module import AdaptiveUpsampling


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# --- Resize positional embeddings ---
def resize_pos_embed(pos_embed, new_grid_size, cls_token=True):
    """
    pos_embed: [1,197,768] or [1, N+1, D]
    new_grid_size: int, e.g. 16, 32
    """
    if cls_token:
        cls_pos = pos_embed[:, :1, :]
        pos_tokens = pos_embed[:, 1:, :]
    else:
        cls_pos = None
        pos_tokens = pos_embed

    # old grid size
    h = w = int(pos_tokens.shape[1] ** 0.5)
    pos_tokens = pos_tokens.reshape(1, h, w, -1).permute(0, 3, 1, 2)

    # interpolate
    pos_tokens = F.interpolate(pos_tokens, size=(new_grid_size, new_grid_size),
                               mode="bicubic", align_corners=False)

    # flatten
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size**2, -1)

    return torch.cat([cls_pos, pos_tokens], dim=1) if cls_token else pos_tokens

class TransUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_ch=64, freeze_vit_layers=8, use_attention_gate=True):
        super().__init__()
        self.use_attention_gate = use_attention_gate
        
        # --- Encoder (3 downsamples) ---
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)  # bottleneck (no pool4!)

        # --- ViT ---
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)

        # replace conv_proj: stride=2 to keep 196 patches when input=28×28
        old_conv = self.vit.conv_proj
        self.vit.conv_proj = nn.Conv2d(
            in_channels=base_ch * 8,  # bottleneck channels
            out_channels=old_conv.out_channels,  # 768
            kernel_size=2,
            stride=2,
        )
        nn.init.kaiming_normal_(self.vit.conv_proj.weight, mode="fan_out", nonlinearity="relu")

        self.vit_head_dim = self.vit.hidden_dim  # 768

        # Save pretrained pos_embed for resizing later
        self.register_buffer("pretrained_pos_embed", self.vit.encoder.pos_embedding.data.clone())

        # freeze_vit_layers
        for i, blk in enumerate(self.vit.encoder.layers):
            if i < freeze_vit_layers:
                for param in blk.parameters():
                    param.requires_grad = False

        # --- Decoder with Adaptive Upsampling and Attention Gates ---
        # 使用AU模块替换原有的转置卷积上采样
        self.up1 = AdaptiveUpsampling(self.vit_head_dim)
        # 添加通道调整层，将AU模块输出调整为所需通道数
        self.channel_adj1 = nn.Conv2d(self.vit_head_dim, base_ch * 4, kernel_size=1)
        
        # 添加注意力门控到跳跃连接
        if use_attention_gate:
            self.att_gate1 = AttentionGate(C_enc=base_ch * 4, C_dec=base_ch * 4, C_mid=32)
            self.dec1 = DoubleConv(base_ch * 8, base_ch * 4)
        else:
            self.dec1 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = AdaptiveUpsampling(base_ch * 4)
        self.channel_adj2 = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=1)
        
        if use_attention_gate:
            self.att_gate2 = AttentionGate(C_enc=base_ch * 2, C_dec=base_ch * 2, C_mid=16)
            self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        else:
            self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up3 = AdaptiveUpsampling(base_ch * 2)
        self.channel_adj3 = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)
        
        if use_attention_gate:
            self.att_gate3 = AttentionGate(C_enc=base_ch, C_dec=base_ch, C_mid=8)
            self.dec3 = DoubleConv(base_ch * 2, base_ch)
        else:
            self.dec3 = DoubleConv(base_ch * 2, base_ch)

        self.final_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        assert H % 16 == 0 and W % 16 == 0, f"Input size must be divisible by 16, got {H}x{W}"

        # --- Encoder ---
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))  # bottleneck

        # --- ViT ---
        B, C, H, W = s4.shape
        x = self.vit.conv_proj(s4)  # [B,768,h,w]
        n_h, n_w = x.shape[2], x.shape[3]
        N = n_h * n_w

        # Resize pos_embed dynamically
        grid_size = int(N ** 0.5)
        pos_embed = resize_pos_embed(self.pretrained_pos_embed, grid_size, cls_token=True)
        self.vit.encoder.pos_embedding = nn.Parameter(pos_embed)

        # flatten
        x = x.flatten(2).transpose(1, 2)  # [B,N,768]
        cls_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder(x)  # Transformer
        x = self.vit.encoder.ln(x)
        x = x[:, 1:]  # 移除cls_token

        # reshape back
        x = x.transpose(1, 2).reshape(B, self.vit_head_dim, grid_size, grid_size)
        x = F.interpolate(x, size=(H, W), mode="bilinear")

        # --- Decoder with Attention Gates ---
        x = self.up1(x)
        x = self.channel_adj1(x)  # 调整通道数
        
        if self.use_attention_gate:
            # 使用注意力门控处理跳跃连接
            s3_attended = self.att_gate1(s3, x)
            x = self.dec1(torch.cat([x, s3_attended], dim=1))
        else:
            x = self.dec1(torch.cat([x, s3], dim=1))
        
        x = self.up2(x)
        x = self.channel_adj2(x)  # 调整通道数
        
        if self.use_attention_gate:
            s2_attended = self.att_gate2(s2, x)
            x = self.dec2(torch.cat([x, s2_attended], dim=1))
        else:
            x = self.dec2(torch.cat([x, s2], dim=1))
        
        x = self.up3(x)
        x = self.channel_adj3(x)  # 调整通道数
        
        if self.use_attention_gate:
            s1_attended = self.att_gate3(s1, x)
            x = self.dec3(torch.cat([x, s1_attended], dim=1))
        else:
            x = self.dec3(torch.cat([x, s1], dim=1))

        return self.final_conv(x)

# 创建带注意力门控的模型
model = TransUNet(in_channels=1, out_channels=3, use_attention_gate=True)

# try different input sizes
for size in [224, 256, 512]:
    x = torch.randn(1, 1, size, size)
    y = model(x)
    print(f"input={size} → output={y.shape}")
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model Parameters:")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Model size: {total_params * 4 / 1024 / 1024: .2f} MB")
    print(f"total_parameters: {total_params}")
    print(f"trainable_parameters: {trainable_params}")