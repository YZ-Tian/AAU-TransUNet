import torch
import torch.nn as nn
import torch.nn.functional as F
from TransUNet_original_freeze_ce_loss import TransUNet

# 测试损失函数
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)  # [B,H,W]
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def dice_loss(pred, target, eps=1e-6):
    pred = torch.softmax(pred, dim=1)  # [B,C,H,W]
    target_onehot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 注意：这里需要调整，因为我们的模型输出是3通道的logits
        # 对于多类别分割，我们需要先转换为概率
        inputs = torch.softmax(inputs, dim=1)  # [B,3,H,W]
        
        # 计算每个类别的Tversky损失
        total_loss = 0
        for cls in range(3):
            # 二值化当前类别
            pred_cls = inputs[:, cls:cls+1, :, :]
            target_cls = (targets == cls).float().unsqueeze(1)
            
            # Flatten
            pred_cls = pred_cls.reshape(-1)
            target_cls = target_cls.reshape(-1)

            TP = (pred_cls * target_cls).sum()
            FP = ((1 - target_cls) * pred_cls).sum()
            FN = (target_cls * (1 - pred_cls)).sum()

            TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            total_loss += (1 - TI) ** self.gamma
        
        return total_loss / 3  # 平均所有类别的损失

def build_loss_fn(device):
    class_weights = torch.tensor([0.2, 0.3, 0.5], device=device)
    focal_loss = FocalLoss(weight=class_weights, gamma=2.0)
    focal_tversky_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)

    def combined_loss(pred, target):
        return 0.3 * dice_loss(pred, target) + 0.3 * focal_loss(pred, target) + 0.4 * focal_tversky_loss(pred, target)

    return combined_loss

def test_loss_functions():
    print("=== 测试损失函数与集成AU模块的TransUNet模型 ===")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模型
    print("\n1. 创建TransUNet模型...")
    model = TransUNet(
        in_channels=1, 
        out_channels=3, 
        base_ch=64,
        freeze_vit_layers=8,
        use_attention_gate=True
    ).to(device)
    
    # 创建测试数据
    print("2. 创建测试数据...")
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 224, 224).to(device)
    target_tensor = torch.randint(0, 3, (batch_size, 224, 224)).to(device)
    
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"目标张量形状: {target_tensor.shape}")
    
    # 前向传播
    print("3. 执行前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        print(f"模型输出形状: {output.shape}")
        print(f"输出数据类型: {output.dtype}")
        print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 测试各个损失函数
    print("\n4. 测试各个损失函数...")
    
    # 测试FocalLoss
    print("   - 测试FocalLoss...")
    focal_loss_fn = FocalLoss(weight=torch.tensor([0.2, 0.3, 0.5], device=device))
    focal_loss_value = focal_loss_fn(output, target_tensor)
    print(f"     FocalLoss值: {focal_loss_value.item():.6f}")
    
    # 测试DiceLoss
    print("   - 测试DiceLoss...")
    dice_loss_value = dice_loss(output, target_tensor)
    print(f"     DiceLoss值: {dice_loss_value.item():.6f}")
    
    # 测试FocalTverskyLoss
    print("   - 测试FocalTverskyLoss...")
    focal_tversky_loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)
    focal_tversky_loss_value = focal_tversky_loss_fn(output, target_tensor)
    print(f"     FocalTverskyLoss值: {focal_tversky_loss_value.item():.6f}")
    
    # 测试组合损失函数
    print("   - 测试组合损失函数...")
    combined_loss_fn = build_loss_fn(device)
    combined_loss_value = combined_loss_fn(output, target_tensor)
    print(f"     组合损失值: {combined_loss_value.item():.6f}")
    
    # 测试梯度计算
    print("\n5. 测试梯度计算...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 前向传播
    output = model(input_tensor)
    loss = combined_loss_fn(output, target_tensor)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"   总梯度范数: {total_grad_norm:.6f}")
    print(f"   损失值: {loss.item():.6f}")
    
    print("\n=== 测试完成 ===")
    print("✅ 所有损失函数都能与集成AU模块的TransUNet模型正常配合使用！")
    
    return True

if __name__ == "__main__":
    test_loss_functions()
