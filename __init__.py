"""
Fielix Architecture - A Novel Neural Network Architecture
Fielix 架构 - 全新的神经网络架构

不同于 Transformer、RNN、CNN、SSM 等现有架构，Fielix 引入了以下创新：

1. 场效应传播 (Field Effect Propagation)
   - 信息像物理场一样在特征空间中传播
   - 替代显式的注意力矩阵计算
   - 复杂度 O(n*k) 而非 O(n²)

2. 动态拓扑连接 (Dynamic Topology Connection)
   - 连接模式根据输入内容动态生成
   - 稀疏且自适应
   - 层次化的多尺度连接

3. 螺旋记忆 (Spiral Memory)
   - 多时间尺度的记忆系统
   - 模拟人脑的记忆巩固过程
   - 支持长期依赖和外部知识

4. 涌现式位置编码 (Emergent Position Encoding)
   - 位置信息从交互中涌现
   - 上下文相关
   - 支持任意长度外推

Author: Fielix Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Fielix Research Team"

from .core import (
    # Field Propagation
    FieldGenerator,
    FieldPropagator,
    FieldSensor,
    FieldEffectLayer,
    
    # Dynamic Topology
    TopologyPredictor,
    SparseMessagePassing,
    HierarchicalTopology,
    DynamicTopologyLayer,
    
    # Spiral Memory
    MemoryCell,
    SpiralMemoryBank,
    SpiralMemoryLayer,
    RecurrentSpiralMemory,
    
    # Emergent Position
    PositionOscillator,
    RelationalPositionLearner,
    EmergentPositionEncoder,
    FlowPositionEncoder,
    PositionExtrapolator,
    
    # Feedforward
    AdaptiveWidthMLP,
    SparseExpertFFN,
    GatedFFN,
    FielixFeedForward,
    
    # Fielix Block
    FielixBlock,
    FielixPreNorm,
    FielixResidual,
    CrossFielixBlock,
    CrossTopologyAttention,
)

from .models import (
    FielixConfig,
    FielixEmbedding,
    FielixDecoder,
    FielixLMHead,
    FielixForCausalLM,
    FielixForSequenceClassification,
    FielixForTokenClassification
)


def create_fielix_tiny() -> 'FielixForCausalLM':
    """创建 Tiny 版本 (~25M 参数)"""
    config = FielixConfig(
        vocab_size=32000,
        dim=256,
        num_layers=6,
        attention_type='field',
        use_memory=False,
        ffn_type='gated'
    )
    return FielixForCausalLM(config)


def create_fielix_small() -> 'FielixForCausalLM':
    """创建 Small 版本 (~125M 参数)"""
    config = FielixConfig(
        vocab_size=32000,
        dim=512,
        num_layers=12,
        attention_type='hybrid',
        use_memory=True,
        ffn_type='gated'
    )
    return FielixForCausalLM(config)


def create_fielix_base() -> 'FielixForCausalLM':
    """创建 Base 版本 (~350M 参数)"""
    config = FielixConfig(
        vocab_size=32000,
        dim=768,
        num_layers=12,
        attention_type='hybrid',
        use_memory=True,
        ffn_type='gated'
    )
    return FielixForCausalLM(config)


def create_fielix_large() -> 'FielixForCausalLM':
    """创建 Large 版本 (~760M 参数)"""
    config = FielixConfig(
        vocab_size=32000,
        dim=1024,
        num_layers=24,
        attention_type='hybrid',
        use_memory=True,
        ffn_type='moe',
        num_experts=8
    )
    return FielixForCausalLM(config)


__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Core components
    'FieldGenerator',
    'FieldPropagator',
    'FieldSensor',
    'FieldEffectLayer',
    'TopologyPredictor',
    'SparseMessagePassing',
    'HierarchicalTopology',
    'DynamicTopologyLayer',
    'MemoryCell',
    'SpiralMemoryBank',
    'SpiralMemoryLayer',
    'RecurrentSpiralMemory',
    'PositionOscillator',
    'RelationalPositionLearner',
    'EmergentPositionEncoder',
    'FlowPositionEncoder',
    'PositionExtrapolator',
    'AdaptiveWidthMLP',
    'SparseExpertFFN',
    'GatedFFN',
    'FielixFeedForward',
    'FielixBlock',
    'FielixPreNorm',
    'FielixResidual',
    'CrossFielixBlock',
    'CrossTopologyAttention',
    
    # Models
    'FielixConfig',
    'FielixEmbedding',
    'FielixDecoder',
    'FielixLMHead',
    'FielixForCausalLM',
    'FielixForSequenceClassification',
    'FielixForTokenClassification',
    
    # Factory functions
    'create_fielix_tiny',
    'create_fielix_small',
    'create_fielix_base',
    'create_fielix_large',
]
