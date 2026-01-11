import torch
import torch.nn as nn
import torch.nn.functional as F
from TransUNet_original_freeze_ce_loss import TransUNet
from utils import get_loaders
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================================================================
# Enhanced TransUNet Validation Script
# 本脚本用于评估集成AU模块、AG模块和高级损失函数的TransUNet模型
# 提供全面的性能评估、可视化分析和模型诊断
# ============================================================================

# 配置参数
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_IMG_DIR = r"E:\python_E\TransUNet_original_freeze_ce_loss_update\dataset_tumor_only\images_validation_tumor_only"
VAL_LABEL_DIR = r"E:\python_E\TransUNet_original_freeze_ce_loss_update\dataset_tumor_only\labels_validation_tumor_only"
MODEL_SAVE_PATH = r"E:\python_E\TransUNet_original_freeze_ce_loss_update\best_model.pth.tar"
RESULTS_SAVE_DIR = r"E:\python_E\TransUNet_original_freeze_ce_loss_update\validation_results"

# 类别名称和颜色映射
CLASS_NAMES = ["Background", "Liver", "Tumor"]
CLASS_COLORS = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]  # 黑、灰、白


def load_model():
    """加载集成AU模块、AG模块的TransUNet模型"""
    print("Loading Enhanced TransUNet model with AU and AG modules...")
    
    # 创建模型
    model = TransUNet(
        in_channels=1, 
        out_channels=3, 
        base_ch=64,
        freeze_vit_layers=8,
        use_attention_gate=True  # 启用注意力门控
    ).to(device=DEVICE)
    
    # 加载检查点
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    
    model_dict = model.state_dict()
    pretrained_dict = checkpoint["state_dict"]
    
    # 过滤掉不匹配shape的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, checkpoint


def calculate_metrics(pred, target, num_classes=3):
    """计算详细的评估指标"""
    metrics = {}
    
    # 转换为numpy数组
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # 计算每个类别的指标
    for cls in range(num_classes):
        pred_cls = (pred_np == cls)
        target_cls = (target_np == cls)
        
        # 计算混淆矩阵元素
        tp = np.sum((pred_cls) & (target_cls))
        fp = np.sum((pred_cls) & (~target_cls))
        fn = np.sum((~pred_cls) & (target_cls))
        tn = np.sum((~pred_cls) & (~target_cls))
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Dice系数
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # IoU
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        # 灵敏度（召回率）
        sensitivity = recall
        
        # 特异性
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics[f'class_{cls}'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'dice': dice,
            'iou': iou,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    return metrics


def plot_confusion_matrix(pred, target, save_path):
    """绘制混淆矩阵"""
    pred_flat = pred.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()
    
    cm = confusion_matrix(target_flat, pred_flat, labels=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_history, save_dir):
    """绘制指标对比图"""
    epochs = list(range(1, len(metrics_history) + 1))
    
    # 提取指标
    dice_scores = np.array([m['mean_dice'] for m in metrics_history])
    iou_scores = np.array([m['mean_iou'] for m in metrics_history])
    
    # 绘制Dice和IoU对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Dice分数
    ax1.plot(epochs, dice_scores[:, 0], 'o-', label='Background', color='black')
    ax1.plot(epochs, dice_scores[:, 1], 's-', label='Liver', color='gray')
    ax1.plot(epochs, dice_scores[:, 2], '^-', label='Tumor', color='red')
    ax1.plot(epochs, dice_scores.mean(axis=1), 'd--', label='mDice', color='blue', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Scores Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # IoU分数
    ax2.plot(epochs, iou_scores[:, 0], 'o-', label='Background', color='black')
    ax2.plot(epochs, iou_scores[:, 1], 's-', label='Liver', color='gray')
    ax2.plot(epochs, iou_scores[:, 2], '^-', label='Tumor', color='red')
    ax2.plot(epochs, iou_scores.mean(axis=1), 'd--', label='mIoU', color='blue', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU Score')
    ax2.set_title('IoU Scores Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def enhanced_accuracy_check(loader, model, device="cuda"):
    """增强的准确性检查，包含详细指标"""
    print("Starting enhanced accuracy evaluation...")
    
    all_metrics = []
    total_time = 0
    total_samples = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            start_time = time.time()
            
            x = x.to(device)
            y = y.to(device)
            
            # 前向传播
            predictions = model(x)
            predicted = torch.argmax(predictions, dim=1)
            
            # 计算时间
            batch_time = time.time() - start_time
            total_time += batch_time
            total_samples += x.size(0)
            
            # 计算详细指标
            batch_metrics = calculate_metrics(predicted, y)
            all_metrics.append(batch_metrics)
            
            # 每10个batch保存可视化结果
            if batch_idx % 10 == 0:
                save_enhanced_results(
                    x[0].cpu(), y[0].cpu(), predictions[0].cpu(), 
                    batch_idx, 0, RESULTS_SAVE_DIR
                )
                
                # 绘制混淆矩阵
                cm_save_path = os.path.join(RESULTS_SAVE_DIR, f'confusion_matrix_batch_{batch_idx}.png')
                plot_confusion_matrix(predicted[0], y[0], cm_save_path)
            
            # 打印进度
            if batch_idx % 5 == 0:
                print(f"Processed batch {batch_idx}/{len(loader)}, "
                      f"Time: {batch_time:.3f}s, "
                      f"Avg Dice: {np.mean([m['class_0']['dice'] for m in batch_metrics]):.4f}")
    
    # 汇总所有指标
    print("\nAggregating metrics...")
    aggregated_metrics = aggregate_metrics(all_metrics)
    
    # 打印详细结果
    print_detailed_results(aggregated_metrics)
    
    # 计算平均推理时间
    avg_inference_time = total_time / total_samples
    print(f"\nAverage inference time per sample: {avg_inference_time:.4f}s")
    
    return aggregated_metrics


def aggregate_metrics(all_metrics):
    """聚合所有batch的指标"""
    aggregated = {}
    
    for cls in range(3):
        cls_metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'dice': [],
            'iou': [],
            'sensitivity': [],
            'specificity': []
        }
        
        for batch_metrics in all_metrics:
            cls_key = f'class_{cls}'
            if cls_key in batch_metrics:
                for metric in cls_metrics:
                    cls_metrics[metric].append(batch_metrics[cls_key][metric])
        
        # 计算平均值和标准差
        aggregated[f'class_{cls}'] = {}
        for metric in cls_metrics:
            values = np.array(cls_metrics[metric])
            aggregated[f'class_{cls}'][f'{metric}_mean'] = np.mean(values)
            aggregated[f'class_{cls}'][f'{metric}_std'] = np.std(values)
    
    # 计算总体指标
    dice_scores = [np.mean([m[f'class_{cls}']['dice'] for m in all_metrics]) for cls in range(3)]
    iou_scores = [np.mean([m[f'class_{cls}']['iou'] for m in all_metrics]) for cls in range(3)]
    
    aggregated['overall'] = {
        'mean_dice': np.array(dice_scores),
        'mean_iou': np.array(iou_scores),
        'mDice': np.mean(dice_scores),
        'mIoU': np.mean(iou_scores)
    }
    
    return aggregated


def print_detailed_results(metrics):
    """打印详细的评估结果"""
    print("\n" + "="*80)
    print("ENHANCED TRANSUUNET MODEL EVALUATION RESULTS")
    print("="*80)
    
    # 打印每个类别的详细指标
    for cls in range(3):
        cls_name = CLASS_NAMES[cls]
        cls_metrics = metrics[f'class_{cls}']
        
        print(f"\n{cls_name.upper()} CLASS METRICS:")
        print(f"  Dice Score:     {cls_metrics['dice_mean']:.4f} ± {cls_metrics['dice_std']:.4f}")
        print(f"  IoU Score:      {cls_metrics['iou_mean']:.4f} ± {cls_metrics['iou_std']:.4f}")
        print(f"  Precision:      {cls_metrics['precision_mean']:.4f} ± {cls_metrics['precision_std']:.4f}")
        print(f"  Recall:         {cls_metrics['recall_mean']:.4f} ± {cls_metrics['recall_std']:.4f}")
        print(f"  F1-Score:       {cls_metrics['f1_score_mean']:.4f} ± {cls_metrics['f1_score_std']:.4f}")
        print(f"  Sensitivity:    {cls_metrics['sensitivity_mean']:.4f} ± {cls_metrics['sensitivity_std']:.4f}")
        print(f"  Specificity:    {cls_metrics['specificity_mean']:.4f} ± {cls_metrics['specificity_std']:.4f}")
    
    # 打印总体指标
    overall = metrics['overall']
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  mDice:           {overall['mDice']:.4f}")
    print(f"  mIoU:            {overall['mIoU']:.4f}")
    print(f"  Background Dice: {overall['mean_dice'][0]:.4f}")
    print(f"  Liver Dice:      {overall['mean_dice'][1]:.4f}")
    print(f"  Tumor Dice:      {overall['mean_dice'][2]:.4f}")
    
    print("="*80)


def save_enhanced_results(image, label, pred, batch_idx, sample_idx, save_dir):
    """保存增强的可视化结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 处理输入数据
    if len(image.shape) == 4:
        image = image[0]
        label = label[0]
        pred = pred[0]
    
    # 转换为numpy数组
    image_np = image.numpy().transpose(1, 2, 0)
    label_np = label.numpy()
    
    # 计算概率图和预测掩码
    prob_map = torch.softmax(pred, dim=0)
    pred_mask_np = torch.argmax(prob_map, dim=0).numpy()
    
    # 归一化图像
    if image_np.max() > 1:
        image_np = image_np / 255.0
    
    # 创建可视化图像
    label_vis = np.zeros_like(label_np, dtype=np.uint8)
    label_vis[label_np == 1] = 128
    label_vis[label_np == 2] = 255
    
    H, W = pred_mask_np.shape
    pred_vis = np.zeros((H, W, 3), dtype=np.uint8)
    pred_vis[pred_mask_np == 1] = [128, 128, 128]
    pred_vis[pred_mask_np == 2] = [255, 255, 255]
    
    # 创建概率图可视化
    prob_vis = np.zeros((H, W, 3), dtype=np.uint8)
    tumor_prob = prob_map[2].numpy()
    prob_vis[:, :, 0] = (tumor_prob * 255).astype(np.uint8)  # 红色通道显示肿瘤概率
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(image_np.squeeze(), cmap='gray')
    axes[0, 0].set_title("Input Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 真实标签
    axes[0, 1].imshow(label_vis, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 预测结果
    axes[1, 0].imshow(pred_vis, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title("Prediction", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 肿瘤概率图
    axes[1, 1].imshow(prob_vis[:, :, 0], cmap='hot', alpha=0.8)
    axes[1, 1].set_title("Tumor Probability", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    filename = f"enhanced_result_batch{batch_idx:03d}_sample{sample_idx:03d}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Enhanced result saved to {save_path}")


def main():
    """主函数"""
    print("="*80)
    print("ENHANCED TRANSUUNET MODEL VALIDATION")
    print("Model: TransUNet + AU Module + AG Module + Advanced Loss Functions")
    print("="*80)
    
    # 创建结果保存目录
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    
    # 加载验证数据
    print("\nLoading validation data...")
    _, val_loader = get_loaders(
        None, None, VAL_IMG_DIR, VAL_LABEL_DIR,
        BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )
    print(f"Validation data loaded: {len(val_loader)} batches")
    
    # 加载模型
    print("\nLoading model...")
    model, checkpoint = load_model()
    
    # 评估模型性能
    print("\nStarting model evaluation...")
    metrics = enhanced_accuracy_check(val_loader, model, DEVICE)
    
    # 保存评估结果
    results_file = os.path.join(RESULTS_SAVE_DIR, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("Enhanced TransUNet Model Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model: TransUNet + AU Module + AG Module\n")
        f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for cls in range(3):
            cls_name = CLASS_NAMES[cls]
            cls_metrics = metrics[f'class_{cls}']
            f.write(f"{cls_name} Class:\n")
            for metric, value in cls_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
        
        overall = metrics['overall']
        f.write(f"Overall Performance:\n")
        f.write(f"  mDice: {overall['mDice']:.4f}\n")
        f.write(f"  mIoU: {overall['mIoU']:.4f}\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Visualization results saved to: {RESULTS_SAVE_DIR}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()