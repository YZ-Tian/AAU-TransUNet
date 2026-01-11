"""
测试 AttentionGate 与 TransUNet 的集成
"""

import torch
import torch.nn as nn
from TransUNet_original_freeze_ce_loss import TransUNet
from AttentionGate import AttentionGate
from config import Config

def test_attention_gate():
    """测试 AttentionGate 模块"""
    print("=== 测试 AttentionGate 模块 ===")
    
    # 测试参数
    batch_size = 2
    channels_enc = 64
    channels_dec = 128
    height, width = 32, 32
    
    # 创建 AttentionGate
    ag = AttentionGate(C_enc=channels_enc, C_dec=channels_dec, C_mid=32)
    
    # 创建测试输入
    x_enc = torch.randn(batch_size, channels_enc, height, width)
    x_dec = torch.randn(batch_size, channels_dec, height, width)
    
    # 前向传播
    with torch.no_grad():
        output = ag(x_enc, x_dec)
    
    print(f"编码器输入形状: {x_enc.shape}")
    print(f"解码器输入形状: {x_dec.shape}")
    print(f"输出形状: {output.shape}")
    print(f"AttentionGate 参数量: {sum(p.numel() for p in ag.parameters()):,}")
    
    # 验证输出形状
    assert output.shape == x_enc.shape, f"输出形状不匹配: {output.shape} vs {x_enc.shape}"
    print("✓ AttentionGate 形状测试通过")
    
    return ag

def test_transunet_with_attention():
    """测试带注意力门控的 TransUNet"""
    print("\n=== 测试带注意力门控的 TransUNet ===")
    
    # 创建模型
    model_with_ag = TransUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_ch=Config.BASE_CH,
        freeze_vit_layers=Config.FREEZE_VIT_LAYERS,
        use_attention_gate=True
    )
    
    model_without_ag = TransUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_ch=Config.BASE_CH,
        freeze_vit_layers=Config.FREEZE_VIT_LAYERS,
        use_attention_gate=False
    )
    
    print(f"模型配置:")
    print(f"  - 输入通道: {Config.IN_CHANNELS}")
    print(f"  - 输出通道: {Config.OUT_CHANNELS}")
    print(f"  - 基础通道: {Config.BASE_CH}")
    print(f"  - 冻结 ViT 层数: {Config.FREEZE_VIT_LAYERS}")
    
    # 计算参数量
    total_params_with_ag = sum(p.numel() for p in model_with_ag.parameters())
    trainable_params_with_ag = sum(p.numel() for p in model_with_ag.parameters() if p.requires_grad)
    
    total_params_without_ag = sum(p.numel() for p in model_without_ag.parameters())
    trainable_params_without_ag = sum(p.numel() for p in model_without_ag.parameters() if p.requires_grad)
    
    print(f"\n参数量对比:")
    print(f"  带注意力门控:")
    print(f"    - 总参数量: {total_params_with_ag:,}")
    print(f"    - 可训练参数量: {trainable_params_with_ag:,}")
    print(f"    - 模型大小: {total_params_with_ag * 4 / 1024 / 1024:.2f} MB")
    
    print(f"  不带注意力门控:")
    print(f"    - 总参数量: {total_params_without_ag:,}")
    print(f"    - 可训练参数量: {trainable_params_without_ag:,}")
    print(f"    - 模型大小: {total_params_without_ag * 4 / 1024 / 1024:.2f} MB")
    
    # 计算增加的参数量
    param_increase = total_params_with_ag - total_params_without_ag
    param_increase_mb = param_increase * 4 / 1024 / 1024
    print(f"\n注意力门控增加的参数量: {param_increase:,} ({param_increase_mb:.2f} MB)")
    
    return model_with_ag, model_without_ag

def test_forward_pass(model, model_name, input_size=256):
    """测试模型前向传播"""
    print(f"\n=== 测试 {model_name} 前向传播 ===")
    
    # 创建测试输入
    x = torch.randn(1, Config.IN_CHANNELS, input_size, input_size)
    
    # 前向传播
    with torch.no_grad():
        y = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # 验证输出形状
    expected_output_shape = (1, Config.OUT_CHANNELS, input_size, input_size)
    assert y.shape == expected_output_shape, f"输出形状不匹配: {y.shape} vs {expected_output_shape}"
    print(f"✓ {model_name} 前向传播测试通过")
    
    return y

def test_different_input_sizes():
    """测试不同输入尺寸"""
    print("\n=== 测试不同输入尺寸 ===")
    
    model = TransUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_ch=Config.BASE_CH,
        use_attention_gate=True
    )
    
    input_sizes = [224, 256, 512]
    
    for size in input_sizes:
        try:
            x = torch.randn(1, Config.IN_CHANNELS, size, size)
            with torch.no_grad():
                y = model(x)
            print(f"输入 {size}x{size} → 输出 {y.shape}")
        except Exception as e:
            print(f"输入 {size}x{size} 失败: {e}")

def main():
    """主测试函数"""
    print("开始测试 AttentionGate 与 TransUNet 的集成...")
    
    try:
        # 测试 AttentionGate 模块
        ag = test_attention_gate()
        
        # 测试带注意力门控的 TransUNet
        model_with_ag, model_without_ag = test_transunet_with_attention()
        
        # 测试前向传播
        test_forward_pass(model_with_ag, "带注意力门控的 TransUNet", 256)
        test_forward_pass(model_without_ag, "不带注意力门控的 TransUNet", 256)
        
        # 测试不同输入尺寸
        test_different_input_sizes()
        
        print("\n🎉 所有测试通过！AttentionGate 已成功集成到 TransUNet 中。")
        
        # 保存测试模型
        torch.save(model_with_ag.state_dict(), 'test_model_with_attention_gate.pth')
        print("✓ 测试模型已保存为 'test_model_with_attention_gate.pth'")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
