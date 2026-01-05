"""
Nexus Architecture - Adaptive Feedforward Module
自适应前馈模块

核心创新：前馈网络不是固定的 MLP，而是根据输入内容动态选择专家和路由
- 动态宽度：根据 token 复杂度调整计算量
- 稀疏激活：只激活相关的神经元
- 知识分片：不同专家存储不同类型的知识
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class AdaptiveWidthMLP(nn.Module):
    """
    自适应宽度 MLP：根据输入复杂度动态调整网络宽度
    
    简单的 token 用窄网络处理，复杂的 token 用宽网络处理
    """
    
    def __init__(
        self,
        dim: int,
        max_hidden_dim: int = None,
        num_width_levels: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_hidden_dim = max_hidden_dim or dim * 4
        self.num_width_levels = num_width_levels
        
        # 宽度级别
        self.width_levels = [
            self.max_hidden_dim // (2 ** (num_width_levels - 1 - i))
            for i in range(num_width_levels)
        ]
        
        # 复杂度预测器
        self.complexity_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_width_levels)
        )
        
        # 共享的输入投影（最大宽度）
        self.input_proj = nn.Linear(dim, self.max_hidden_dim)
        
        # 共享的输出投影
        self.output_proj = nn.Linear(self.max_hidden_dim, dim)
        
        # 激活函数
        self.activation = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 宽度掩码（用于稀疏激活）
        self._init_width_masks()
    
    def _init_width_masks(self):
        """初始化宽度掩码"""
        masks = []
        for width in self.width_levels:
            mask = torch.zeros(self.max_hidden_dim)
            mask[:width] = 1.0
            masks.append(mask)
        self.register_buffer('width_masks', torch.stack(masks))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        
        Returns:
            output: (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape
        
        # 预测复杂度
        complexity_logits = self.complexity_predictor(x)  # (batch, seq_len, num_levels)
        complexity_weights = F.softmax(complexity_logits, dim=-1)
        
        # 输入投影
        hidden = self.input_proj(x)  # (batch, seq_len, max_hidden_dim)
        hidden = self.activation(hidden)
        
        # 计算加权掩码
        # (batch, seq_len, num_levels) @ (num_levels, max_hidden_dim) -> (batch, seq_len, max_hidden_dim)
        effective_mask = torch.matmul(complexity_weights, self.width_masks)
        
        # 应用掩码（软性稀疏）
        hidden = hidden * effective_mask
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # 输出投影
        output = self.output_proj(hidden)
        
        return output


class SparseExpertFFN(nn.Module):
    """
    稀疏专家前馈网络：混合专家 (MoE) 的变体
    
    与标准 MoE 不同：
    - 使用软路由而非 top-k 硬路由
    - 专家可以重叠激活
    - 支持专家间的信息共享
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        num_experts: int = 8,
        expert_capacity: float = 1.25,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.hidden_dim // num_experts),
                nn.GELU(),
                nn.Linear(self.hidden_dim // num_experts, dim)
            )
            for _ in range(num_experts)
        ])
        
        # 路由器
        self.router = nn.Linear(dim, num_experts)
        
        # 专家间共享层（促进知识迁移）
        self.shared_layer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        
        # 负载平衡损失系数
        self.load_balance_weight = 0.01
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, dim)
            return_aux_loss: 是否返回负载平衡损失
        
        Returns:
            output: (batch, seq_len, dim)
            aux_loss: 可选的辅助损失
        """
        batch, seq_len, dim = x.shape
        
        # 计算路由权重
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_weights = F.softmax(router_logits, dim=-1)
        
        # 计算共享表示
        shared_repr = self.shared_layer(x)
        
        # 专家输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        
        # 堆叠专家输出
        expert_outputs = torch.stack(expert_outputs, dim=2)  # (batch, seq_len, num_experts, dim)
        
        # 加权求和
        weighted_output = (expert_outputs * router_weights.unsqueeze(-1)).sum(dim=2)
        
        # 添加共享表示
        output = weighted_output + shared_repr * 0.1
        
        output = self.dropout(output)
        
        # 计算负载平衡损失
        aux_loss = None
        if return_aux_loss:
            # 鼓励均匀的专家使用
            expert_usage = router_weights.mean(dim=[0, 1])  # (num_experts,)
            uniform_target = torch.ones_like(expert_usage) / self.num_experts
            aux_loss = F.mse_loss(expert_usage, uniform_target) * self.load_balance_weight
        
        return output, aux_loss


class GatedFFN(nn.Module):
    """
    门控前馈网络：使用 GLU 变体
    
    灵感来自 SwiGLU，但增加了动态门控
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * 8 / 3)  # SwiGLU 的最优比例
        
        # 门控投影
        self.gate_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        
        # 值投影
        self.up_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        
        # 输出投影
        self.down_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        
        # 动态门控调制
        self.gate_modulator = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        
        Returns:
            output: (batch, seq_len, dim)
        """
        # 门控
        gate = F.silu(self.gate_proj(x))
        
        # 值
        up = self.up_proj(x)
        
        # 动态调制
        modulation = self.gate_modulator(x)
        
        # GLU 操作
        hidden = gate * up * modulation
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # 输出
        output = self.down_proj(hidden)
        
        return output


class NexusFeedForward(nn.Module):
    """
    Nexus 前馈层：组合多种前馈机制
    
    根据配置选择：
    - 自适应宽度 MLP
    - 稀疏专家网络
    - 门控 FFN
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        ffn_type: str = 'gated',  # 'adaptive', 'moe', 'gated'
        num_experts: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.ffn_type = ffn_type
        
        # 输入归一化
        self.norm = nn.LayerNorm(dim)
        
        # 选择前馈类型
        if ffn_type == 'adaptive':
            self.ffn = AdaptiveWidthMLP(dim, hidden_dim, dropout=dropout)
        elif ffn_type == 'moe':
            self.ffn = SparseExpertFFN(dim, hidden_dim, num_experts, dropout=dropout)
        else:  # gated
            self.ffn = GatedFFN(dim, hidden_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, dim)
        
        Returns:
            output: (batch, seq_len, dim)
            aux_loss: 可选的辅助损失（仅 MoE）
        """
        residual = x
        x = self.norm(x)
        
        aux_loss = None
        if self.ffn_type == 'moe':
            ffn_out, aux_loss = self.ffn(x, return_aux_loss=True)
        else:
            ffn_out = self.ffn(x)
        
        ffn_out = self.dropout(ffn_out)
        output = residual + ffn_out
        
        return output, aux_loss
