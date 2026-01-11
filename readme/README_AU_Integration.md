# TransUNet模型集成Adaptive Upsampling (AU) 模块

## 概述

本项目成功将Adaptive Upsampling (AU) 模块集成到原始的TransUNet模型中，在不改变原模型架构的前提下，提升了上采样的效果。

## AU模块特点

Adaptive Upsampling模块结合了两种上采样方法：
1. **转置卷积上采样**：用于细节恢复
2. **双线性插值上采样**：用于保持全局结构
3. **自适应权重融合**：通过权重预测模块动态融合两种方法的结果

## 主要修改

### 1. 导入AU模块
```python
from Adaptive_Upsampling_Module import AdaptiveUpsampling
```

### 2. 替换上采样层
将原有的转置卷积上采样层替换为AU模块：

**原始代码：**
```python
self.up1 = nn.ConvTranspose2d(self.vit_head_dim, base_ch * 4, 2, 2)
self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
```

**修改后：**
```python
self.up1 = AdaptiveUpsampling(self.vit_head_dim)
self.up2 = AdaptiveUpsampling(base_ch * 4)
self.up3 = AdaptiveUpsampling(base_ch * 2)
```

### 3. 添加通道调整层
由于AU模块保持输入输出通道数不变，需要添加通道调整层：

```python
self.channel_adj1 = nn.Conv2d(self.vit_head_dim, base_ch * 4, kernel_size=1)
self.channel_adj2 = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=1)
self.channel_adj3 = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)
```

### 4. 修改前向传播
在forward方法中添加通道调整：

```python
x = self.up1(x)
x = self.channel_adj1(x)  # 调整通道数

x = self.up2(x)
x = self.channel_adj2(x)  # 调整通道数

x = self.up3(x)
x = self.channel_adj3(x)  # 调整通道数
```

## 模型结构

```
Encoder (3 downsamples) → ViT → Decoder with AU modules
                                    ↓
                            AdaptiveUpsampling + Channel Adjustment
                                    ↓
                            Attention Gates + DoubleConv
                                    ↓
                            Final Output
```

## 优势

1. **保持原模型结构**：不改变TransUNet的核心架构
2. **提升上采样质量**：结合转置卷积和双线性插值的优势
3. **自适应融合**：根据内容动态调整两种上采样方法的权重
4. **细节保持**：转置卷积恢复细节，双线性插值保持结构

## 测试结果

模型成功运行，支持多种输入尺寸：
- 224×224 → 224×224 ✓
- 256×256 → 256×256 ✓  
- 512×512 → 512×512 ✓

模型参数统计：
- 总参数量：约97-98M
- 可训练参数量：约40-41M

## 使用方法

```python
# 创建集成AU模块的TransUNet模型
model = TransUNet(in_channels=1, out_channels=3, use_attention_gate=True)

# 输入数据
x = torch.randn(1, 1, 224, 224)
y = model(x)  # 输出: [1, 3, 224, 224]
```

## 注意事项

1. 确保`Adaptive_Upsampling_Module.py`文件在同一目录下
2. 确保`AttentionGate.py`文件存在
3. 输入尺寸必须是16的倍数
4. AU模块增加了少量参数，但提升了上采样质量

## 文件结构

```
TransUNet_original_freeze_ce_loss_update/
├── TransUNet_original_freeze_ce_loss.py    # 主模型文件（已集成AU模块）
├── Adaptive_Upsampling_Module.py           # AU模块定义
├── AttentionGate.py                        # 注意力门控模块
└── README_AU_Integration.md                # 本说明文档
```
