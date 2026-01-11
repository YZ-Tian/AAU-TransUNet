import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from TransUNet_original_freeze_ce_loss import TransUNet
from utils import get_loaders
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ============================================================================
# TransUNet with AttentionGate Training Script
# 本脚本使用集成了注意力门控的 TransUNet 模型进行训练
# 注意力门控模块会自动在跳跃连接中工作，增强特征融合效果
# ============================================================================

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCH = 100
NUM_WORKERS = 2
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

TRAIN_IMG_DIR = r"E:\python_E\TransUNet_original_freeze_ce_loss_update\dataset_tumor_only\images_train_tumor_only"
TRAIN_LABEL_DIR = r"E:\python_E\TransUNet_original_freeze_ce_loss_update\dataset_tumor_only\labels_train_tumor_only"
VAL_IMG_DIR = r"E:\python_E\TransUNet_original_freeze_ce_loss_update\dataset_tumor_only\images_validation_tumor_only"
VAL_LABEL_DIR = r"E:\python_E\TransUNet_original_freeze_ce_loss_update\dataset_tumor_only\labels_validation_tumor_only"

MODEL_SAVE_PATH = r"E:\python_E\TransUNet_original_freeze_ce_loss_update\best_model.pth.tar"


class FocalLoss(nn.Module):
    def __init__(self,weight=None,gamma=2.0,reduction="mean"):
        """
              Focal Loss 实现
              Args:
                  weight: 类别权重张量，如 torch.tensor([0.2, 0.3, 0.5])
                  gamma: 调节因子，控制难易样本权重
                  reduction: 损失缩减方式
              """
        super(FocalLoss,self).__init__()
        self.weight=weight
        self.gamma=gamma
        self.reduction=reduction
        self.ce=nn.CrossEntropyLoss(reduction='none',weight=weight)

    def forward(self,inputs,targets):
        ce_loss=self.ce(inputs,targets) #[B,H,W]
        pt=torch.exp(-ce_loss)
        focal_loss=(1-pt)**self.gamma*ce_loss

        if self.reduction=="mean":
            return focal_loss.mean()
        elif self.reduction=="sum":
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
        # inputs: (N, 1, H, W) sigmoid probs
        # targets: (N, 1, H, W) binary mask
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - TI) ** self.gamma

def build_loss_fn(device):
    class_weights = torch.tensor([0.2, 0.3, 0.5], device=device)
    focal_loss = FocalLoss(weight=class_weights, gamma=2.0)
    focal_tversky_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)

    def combined_loss(pred, target):
        return 0.3 * dice_loss(pred, target) + 0.3 * focal_loss(pred, target) + 0.4 * focal_tversky_loss(pred, target)

    return combined_loss


def save_checkpoint(state, filename=MODEL_SAVE_PATH):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def plot_train_loss(train_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_loss_curve.png")
    plt.close()

def train_fn(loader, model, loss_fn, optimizer, scaler, epoch, writer):
    loop = tqdm(loader)
    total_loss = 0.0
    model.train()

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.long().to(DEVICE)

        with torch.cuda.amp.autocast():
            predict = model(data)
            loss = loss_fn(predict, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # 记录batch损失到TensorBoard
        writer.add_scalar('Train/Loss_batch', loss.item(), epoch * len(loader) + batch_idx)
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def check_accuracy(loader, model, device="cuda"):
    dice_scores_all = []
    iou_scores_all = []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)
            predicted = torch.argmax(predictions, dim=1)

            # 计算Dice分数
            dice_scores = []
            for cls in range(3):
                pred_cls = (predicted == cls).float()
                target_cls = (y == cls).float()
                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum()
                dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
                dice_scores.append(dice.item())
            dice_scores_all.append(dice_scores)

            # 计算IoU分数
            iou_scores = []
            for cls in range(3):
                intersection = ((predicted == cls) & (y == cls)).sum().float()
                union = ((predicted == cls) | (y == cls)).sum().float()
                iou = (intersection + 1e-6) / (union + 1e-6)
                iou_scores.append(iou.item())
            iou_scores_all.append(iou_scores)

    dice_scores_all = np.array(dice_scores_all)
    mean_dice = dice_scores_all.mean(axis=0)
    mdice = mean_dice.mean()

    iou_scores_all = np.array(iou_scores_all)
    mean_iou = iou_scores_all.mean(axis=0)
    miou = mean_iou.mean()

    return mean_dice, mean_iou, miou, mdice


def main():
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir='runs/transunet_experiment')

    # 设置随机种子
    seed = random.randint(1, 100)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载数据
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_LABEL_DIR, VAL_IMG_DIR, VAL_LABEL_DIR,
        BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )

    # 初始化模型 - 使用带注意力门控的 TransUNet
    model = TransUNet(
        in_channels=1, 
        out_channels=3, 
        base_ch=64,
        freeze_vit_layers=8,
        use_attention_gate=True  # 明确启用注意力门控
    ).to(device=DEVICE)
    loss_fn = build_loss_fn(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    best_mdice = 0.0
    train_losses = []

    for epoch in range(NUM_EPOCH):
        print(f"Current Epoch: {epoch}")

        # 训练
        train_loss = train_fn(train_loader, model, loss_fn, optimizer, scaler, epoch, writer)
        train_losses.append(train_loss)
        writer.add_scalar('Train/Loss_epoch', train_loss, epoch)

        # 验证并保存最佳模型
        mean_dice, mean_iou, miou, mdice= check_accuracy(val_loader, model, DEVICE)
        writer.add_scalar('Val/mIOU', miou, epoch)

        if mdice > best_mdice:
            best_mdice = mdice
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_mdice": best_mdice,
                "train_losses": train_losses,
            }
            save_checkpoint(checkpoint)

    # 绘制训练损失曲线
    plot_train_loss(train_losses)
    writer.close()


if __name__ == "__main__":
    main()