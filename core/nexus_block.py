"""
Fielix Architecture - Fielix Block
Fielix 块：将所有核心组件组合成一个完整的层

架构设计：
┌─────────────────────────────────────────────────────┐
│                    Input (x)                        │
├─────────────────────────────────────────────────────┤
│              Emergent Position Encoding             │
├─────────────────────────────────────────────────────┤
│                                                     │
│   ┌─────────────┐    ┌─────────────────────────┐   │
│   │ Field Effect│    │  Dynamic Topology       │   │
│   │ Propagation │ OR │  Connection             │   │
│   └─────────────┘    └─────────────────────────┘   │
│                                                     │
├─────────────────────────────────────────────────────┤
│                 Spiral Memory (Optional)            │
├─────────────────────────────────────────────────────┤
│                 Fielix Feed Forward                  │
├─────────────────────────────────────────────────────┤
│                    Output                           │
└─────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

from .field_propagation import FieldEffectLayer
from .dynamic_topology import DynamicTopologyLayer
from .spiral_memory import SpiralMemoryLayer
from .emergent_position import EmergentPositionEncoder
from .feedforward import FielixFeedForward


class FielixBlock(nn.Module):
    """
    Fielix 块：Fielix 架构的基本构建单元
    
    特点：
    - 灵活选择注意力机制（场效应 vs 动态拓扑）
    - 可选的螺旋记忆
    - 自适应前馈网络
    - 涌现式位置编码
    """
    
    def __init__(
        self,
        dim: int,
        attention_type: str = 'field',  # 'field', 'topology', 'hybrid', 'alternating'
        use_memory: bool = False,
        ffn_type: str = 'gated',
        num_experts: int = 8,
        field_iterations: int = 4,
        topology_levels: int = 3,
        memory_levels: int = 3,
        dropout: float = 0.1,
        layer_idx: int = 0  # 用于交替模式
    ):
        super().__init__()
        self.dim = dim
        self.attention_type = attention_type
        self.use_memory = use_memory
        self.layer_idx = layer_idx
        
        # 选择注意力类型
        if attention_type == 'field':
            self.attention = FieldEffectLayer(
                dim,
                num_iterations=field_iterations,
                dropout=dropout
            )
        elif attention_type == 'topology':
            self.attention = DynamicTopologyLayer(
                dim,
                use_hierarchical=True,
                num_levels=topology_levels,
                dropout=dropout
            )
        elif attention_type == 'hybrid':
            # 真正的双层注意力：同时运行 Field + Topology
            self.field_attention = FieldEffectLayer(
                dim,
                num_iterations=field_iterations,
                dropout=dropout
            )
            self.topology_attention = DynamicTopologyLayer(
                dim,
                use_hierarchical=True,
                num_levels=topology_levels,
                dropout=dropout
            )
            # 混合门控
            self.hybrid_gate = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 2),
                nn.Softmax(dim=-1)
            )
        else:  # alternating（显式交替模式）
            if layer_idx % 2 == 0:
                self.attention = FieldEffectLayer(
                    dim,
                    num_iterations=field_iterations,
                    dropout=dropout
                )
                self._attn_type = 'field'
            else:
                self.attention = DynamicTopologyLayer(
                    dim,
                    use_hierarchical=True,
                    num_levels=topology_levels,
                    dropout=dropout
                )
                self._attn_type = 'topology'
        
        # 螺旋记忆（可选）
        if use_memory:
            self.memory = SpiralMemoryLayer(
                dim,
                num_levels=memory_levels,
                dropout=dropout
            )
        
        # 前馈网络
        self.ffn = FielixFeedForward(
            dim,
            ffn_type=ffn_type,
            num_experts=num_experts,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        memory_state: Optional[List[torch.Tensor]] = None,
        causal: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) padding mask
            memory_state: 螺旋记忆状态
            causal: 是否因果
        
        Returns:
            output: (batch, seq_len, dim)
            new_memory_state: 更新后的记忆状态
            aux_loss: 辅助损失
        """
        # 1. 注意力层
        if self.attention_type == 'hybrid':
            # 双层注意力：同时运行 Field + Topology，门控混合
            gate = self.hybrid_gate(x.mean(dim=1, keepdim=True))  # (batch, 1, 2)
            field_out = self.field_attention(x, mask)
            topo_out = self.topology_attention(x, mask, causal)
            x = gate[:, :, 0:1] * field_out + gate[:, :, 1:2] * topo_out
        elif self.attention_type == 'alternating':
            # 交替注意力：每层只运行一种
            if self._attn_type == 'field':
                x = self.attention(x, mask)
            else:
                x = self.attention(x, mask, causal)
        elif self.attention_type == 'field':
            x = self.attention(x, mask)
        else:  # topology
            x = self.attention(x, mask, causal)
        
        # 2. 螺旋记忆（可选）
        new_memory_state = None
        if self.use_memory:
            x, new_memory_state = self.memory(
                x, 
                memories=memory_state,
                return_memories=True
            )
        
        # 3. 前馈网络
        x, aux_loss = self.ffn(x)
        
        return x, new_memory_state, aux_loss


class FielixPreNorm(nn.Module):
    """
    Pre-Normalization 包装器
    """
    
    def __init__(self, dim: int, module: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module
    
    def forward(self, x: torch.Tensor, **kwargs):
        return self.module(self.norm(x), **kwargs)


class FielixResidual(nn.Module):
    """
    带缩放的残差连接
    """
    
    def __init__(self, dim: int, module: nn.Module, scale: float = 1.0):
        super().__init__()
        self.module = module
        self.scale = nn.Parameter(torch.ones(dim) * scale)
    
    def forward(self, x: torch.Tensor, **kwargs):
        output = self.module(x, **kwargs)
        if isinstance(output, tuple):
            return (x + output[0] * self.scale,) + output[1:]
        return x + output * self.scale


class CrossFielixBlock(nn.Module):
    """
    交叉 Fielix 块：用于编码器-解码器架构
    
    支持编码器和解码器之间的信息交换
    """
    
    def __init__(
        self,
        dim: int,
        attention_type: str = 'topology',
        ffn_type: str = 'gated',
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        
        # 自注意力
        self.self_attention = DynamicTopologyLayer(
            dim,
            use_hierarchical=False,
            dropout=dropout
        )
        
        # 交叉注意力
        self.cross_attention = CrossTopologyAttention(dim, dropout=dropout)
        
        # 前馈
        self.ffn = FielixFeedForward(dim, ffn_type=ffn_type, dropout=dropout)
        
        # 归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, tgt_len, dim) 解码器输入
            encoder_output: (batch, src_len, dim) 编码器输出
            self_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
        
        Returns:
            output: (batch, tgt_len, dim)
            aux_loss: 辅助损失
        """
        # 自注意力
        x = x + self.self_attention(self.norm1(x), self_mask, causal=True)
        
        # 交叉注意力
        x = x + self.cross_attention(self.norm2(x), encoder_output, cross_mask)
        
        # 前馈
        ffn_out, aux_loss = self.ffn(self.norm3(x))
        x = x + ffn_out
        
        return x, aux_loss


class CrossTopologyAttention(nn.Module):
    """
    交叉拓扑注意力：用于解码器关注编码器
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q 来自解码器
        self.q_proj = nn.Linear(dim, dim)
        # K, V 来自编码器
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, tgt_len, dim) 解码器查询
            key_value: (batch, src_len, dim) 编码器输出
            mask: 注意力掩码
        
        Returns:
            output: (batch, tgt_len, dim)
        """
        batch, tgt_len, dim = query.shape
        src_len = key_value.shape[1]
        
        # 投影
        q = self.q_proj(query).view(batch, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch, src_len, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch, src_len, self.num_heads, self.head_dim)
        
        # 转置
        q = q.transpose(1, 2)  # (batch, heads, tgt_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, tgt_len, dim)
        
        return self.out_proj(out)
