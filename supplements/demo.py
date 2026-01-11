"""
TransUNet with AttentionGate 演示脚本
展示如何使用优化后的注意力门控模块
"""

import torch
import torch.nn.functional as F
from TransUNet_original_freeze_ce_loss import TransUNet
from AttentionGate import AttentionGate
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_maps(model, input_tensor, save_path=None):
    """可视化注意力掩码"""
    model.eval()
    
    with torch.no_grad():
        # 获取中间特征
        # 这里我们需要修改模型来获取中间特征，暂时跳过
        output = model(input_tensor)
    
    # 创建示例注意力掩码（随机生成用于演示）
    batch_size, channels, height, width = input_tensor.shape
    attention_maps = torch.rand(batch_size, 1, height, width)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始输入
    input_img = input_tensor[0, 0].cpu().numpy()
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # 注意力掩码
    attention_map = attention_maps[0, 0].cpu().numpy()
    axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    # 加权后的特征
    weighted_feature = (input_img * attention_map)
    axes[2].imshow(weighted_feature, cmap='gray')
    axes[2].set_title('Weighted Feature')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力可视化已保存到: {save_path}")
    
    plt.show()

def compare_models():
    """比较带和不带注意力门控的模型"""
    print("=== 模型对比分析 ===")
    
    # 创建模型
    model_with_ag = TransUNet(
        in_channels=1,
        out_channels=3,
        base_ch=64,
        use_attention_gate=True
    )
    
    model_without_ag = TransUNet(
        in_channels=1,
        out_channels=3,
        base_ch=64,
        use_attention_gate=False
    )
    
    # 计算参数量
    params_with_ag = sum(p.numel() for p in model_with_ag.parameters())
    params_without_ag = sum(p.numel() for p in model_without_ag.parameters())
    
    print(f"带注意力门控的模型参数量: {params_with_ag:,}")
    print(f"不带注意力门控的模型参数量: {params_without_ag:,}")
    print(f"增加的参数量: {params_with_ag - params_without_ag:,}")
    print(f"增加比例: {((params_with_ag - params_without_ag) / params_without_ag * 100):.2f}%")
    
    return model_with_ag, model_without_ag

def test_attention_gate_standalone():
    """独立测试 AttentionGate 模块"""
    print("\n=== 独立测试 AttentionGate ===")
    
    # 创建测试数据
    batch_size = 2
    enc_channels = 64
    dec_channels = 128
    height, width = 32, 32
    
    # 创建 AttentionGate
    ag = AttentionGate(
        C_enc=enc_channels,
        C_dec=dec_channels,
        C_mid=32,
        use_bn=True,
        use_residual=True
    )
    
    # 创建测试输入
    x_enc = torch.randn(batch_size, enc_channels, height, width)
    x_dec = torch.randn(batch_size, dec_channels, height, width)
    
    # 前向传播
    with torch.no_grad():
        output = ag(x_enc, x_dec)
    
    print(f"编码器特征形状: {x_enc.shape}")
    print(f"解码器特征形状: {x_dec.shape}")
    print(f"输出特征形状: {output.shape}")
    print(f"AttentionGate 参数量: {sum(p.numel() for p in ag.parameters()):,}")
    
    # 验证输出
    assert output.shape == x_enc.shape, "输出形状不匹配"
    print("✓ AttentionGate 测试通过")
    
    return ag

def performance_test():
    """性能测试"""
    print("\n=== 性能测试 ===")
    
    # 创建模型
    model = TransUNet(
        in_channels=1,
        out_channels=3,
        base_ch=64,
        use_attention_gate=True
    )
    
    # 测试不同输入尺寸的性能
    input_sizes = [128, 256, 512]
    batch_size = 1
    
    for size in input_sizes:
        input_tensor = torch.randn(batch_size, 1, size, size)
        
        # 预热
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 性能测试
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print(f"输入尺寸 {size}x{size}: 平均推理时间 {avg_time:.4f}s")
    
    return model

def main():
    """主函数"""
    print("🚀 TransUNet with AttentionGate 演示")
    print("=" * 50)
    
    try:
        # 1. 模型对比
        model_with_ag, model_without_ag = compare_models()
        
        # 2. 独立测试 AttentionGate
        ag = test_attention_gate_standalone()
        
        # 3. 性能测试
        model = performance_test()
        
        # 4. 创建示例输入并测试
        print("\n=== 示例推理测试 ===")
        input_tensor = torch.randn(1, 1, 256, 256)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"输入形状: {input_tensor.shape}")
        print(f"输出形状: {output.shape}")
        print("✓ 推理测试通过")
        
        # 5. 保存模型
        torch.save(model.state_dict(), 'demo_model.pth')
        print("✓ 演示模型已保存为 'demo_model.pth'")
        
        print("\n🎉 演示完成！")
        print("\n使用说明:")
        print("1. 创建模型: model = TransUNet(use_attention_gate=True)")
        print("2. 训练模型: 使用原有的训练脚本")
        print("3. 推理: output = model(input_tensor)")
        print("4. 注意力门控会自动在跳跃连接中工作")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
