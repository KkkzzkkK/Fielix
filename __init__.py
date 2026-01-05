"""
Nexus Architecture - A Novel Neural Network Architecture
Nexus 架构 - 全新的神经网络架构

不同于 Transformer、RNN、CNN、SSM 等现有架构，Nexus 引入了以下创新：

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

Author: Nexus Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Nexus Research Team"

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
    NexusFeedForward,
    
    # Nexus Block
    NexusBlock,
    NexusPreNorm,
    NexusResidual,
    CrossNexusBlock,
    CrossTopologyAttention,
)

from .models import (
    NexusConfig,
    NexusEmbedding,
    NexusDecoder,
    NexusLMHead,
    NexusForCausalLM,
    NexusForSequenceClassification,
    NexusForTokenClassification
)


def create_nexus_tiny() -> 'NexusForCausalLM':
    """创建 Tiny 版本 (~25M 参数)"""
    config = NexusConfig(
        vocab_size=32000,
        dim=256,
        num_layers=6,
        attention_type='field',
        use_memory=False,
        ffn_type='gated'
    )
    return NexusForCausalLM(config)


def create_nexus_small() -> 'NexusForCausalLM':
    """创建 Small 版本 (~125M 参数)"""
    config = NexusConfig(
        vocab_size=32000,
        dim=512,
        num_layers=12,
        attention_type='hybrid',
        use_memory=True,
        ffn_type='gated'
    )
    return NexusForCausalLM(config)


def create_nexus_base() -> 'NexusForCausalLM':
    """创建 Base 版本 (~350M 参数)"""
    config = NexusConfig(
        vocab_size=32000,
        dim=768,
        num_layers=12,
        attention_type='hybrid',
        use_memory=True,
        ffn_type='gated'
    )
    return NexusForCausalLM(config)


def create_nexus_large() -> 'NexusForCausalLM':
    """创建 Large 版本 (~760M 参数)"""
    config = NexusConfig(
        vocab_size=32000,
        dim=1024,
        num_layers=24,
        attention_type='hybrid',
        use_memory=True,
        ffn_type='moe',
        num_experts=8
    )
    return NexusForCausalLM(config)


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
    'NexusFeedForward',
    'NexusBlock',
    'NexusPreNorm',
    'NexusResidual',
    'CrossNexusBlock',
    'CrossTopologyAttention',
    
    # Models
    'NexusConfig',
    'NexusEmbedding',
    'NexusDecoder',
    'NexusLMHead',
    'NexusForCausalLM',
    'NexusForSequenceClassification',
    'NexusForTokenClassification',
    
    # Factory functions
    'create_nexus_tiny',
    'create_nexus_small',
    'create_nexus_base',
    'create_nexus_large',
]
