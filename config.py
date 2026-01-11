"""
TransUNet with AttentionGate Configuration
"""

class Config:
    # 模型基本参数
    IN_CHANNELS = 1
    OUT_CHANNELS = 3
    BASE_CH = 64
    FREEZE_VIT_LAYERS = 8
    
    # 注意力门控参数
    USE_ATTENTION_GATE = True
    ATTENTION_GATE_CONFIG = {
        'use_bn': True,           # 是否使用批归一化
        'use_residual': True,     # 是否使用残差连接
        'C_mid_scale': 0.5,      # 中间通道数的缩放因子
    }
    
    # 训练参数
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    
    # 数据参数
    IMAGE_SIZE = 256
    NUM_WORKERS = 4
    
    # 模型保存
    SAVE_DIR = './checkpoints'
    MODEL_NAME = 'transunet_with_attention_gate'
    
    @classmethod
    def get_attention_gate_params(cls, enc_channels, dec_channels):
        """根据编码器和解码器通道数计算注意力门控参数"""
        C_mid = int(max(enc_channels, dec_channels) * cls.ATTENTION_GATE_CONFIG['C_mid_scale'])
        C_mid = max(C_mid, 8)  # 最小中间通道数
        
        return {
            'C_enc': enc_channels,
            'C_dec': dec_channels,
            'C_mid': C_mid,
            'use_bn': cls.ATTENTION_GATE_CONFIG['use_bn'],
            'use_residual': cls.ATTENTION_GATE_CONFIG['use_residual']
        }
