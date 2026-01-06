"""
Fielix Architecture - Core Module
Fielix 架构核心模块

包含所有核心组件：
- 场效应传播 (Field Effect Propagation)
- 动态拓扑连接 (Dynamic Topology)
- 螺旋记忆 (Spiral Memory)
- 涌现式位置编码 (Emergent Position)
- 自适应前馈网络 (Adaptive Feedforward)
- Fielix 块 (Fielix Block)
"""

from .field_propagation import (
    FieldPropagator,
    FieldSensor,
    FieldEffectLayer
)

from .dynamic_topology import (
    TopologyPredictor,
    SparseMessagePassing,
    HierarchicalTopology,
    DynamicTopologyLayer
)

from .spiral_memory import (
    MemoryCell,
    SpiralMemoryBank,
    SpiralMemoryLayer,
    RecurrentSpiralMemory
)

from .emergent_position import (
    PositionOscillator,
    RelationalPositionLearner,
    EmergentPositionEncoder,
    FlowPositionEncoder,
    PositionExtrapolator
)

from .feedforward import (
    AdaptiveWidthMLP,
    SparseExpertFFN,
    GatedFFN,
    FielixFeedForward
)

from .nexus_block import (
    FielixBlock,
    FielixPreNorm,
    FielixResidual,
    CrossFielixBlock,
    CrossTopologyAttention
)

from .utils import (
    init_weights,
    init_fielix_model,
    stable_softmax,
    stable_log_softmax,
    clamp_gradients,
    GradientClipping,
    NumericallyStableLayerNorm,
    RMSNorm,
    CheckpointedModule,
    apply_gradient_checkpointing,
    clear_memory,
    MemoryEfficientAttention,
    EMA,
    WarmupCosineScheduler
)

__all__ = [
    # Field Propagation
    'FieldPropagator',
    'FieldSensor',
    'FieldEffectLayer',
    
    # Dynamic Topology
    'TopologyPredictor',
    'SparseMessagePassing',
    'HierarchicalTopology',
    'DynamicTopologyLayer',
    
    # Spiral Memory
    'MemoryCell',
    'SpiralMemoryBank',
    'SpiralMemoryLayer',
    'RecurrentSpiralMemory',
    
    # Emergent Position
    'PositionOscillator',
    'RelationalPositionLearner',
    'EmergentPositionEncoder',
    'FlowPositionEncoder',
    'PositionExtrapolator',
    
    # Feedforward
    'AdaptiveWidthMLP',
    'SparseExpertFFN',
    'GatedFFN',
    'FielixFeedForward',
    
    # Fielix Block
    'FielixBlock',
    'FielixPreNorm',
    'FielixResidual',
    'CrossFielixBlock',
    'CrossTopologyAttention',
    
    # Utils
    'init_weights',
    'init_fielix_model',
    'stable_softmax',
    'stable_log_softmax',
    'clamp_gradients',
    'GradientClipping',
    'NumericallyStableLayerNorm',
    'RMSNorm',
    'CheckpointedModule',
    'apply_gradient_checkpointing',
    'clear_memory',
    'MemoryEfficientAttention',
    'EMA',
    'WarmupCosineScheduler',
]
