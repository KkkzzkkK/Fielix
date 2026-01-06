"""
Fielix Architecture - Field Effect Propagation Module
场效应传播模块 (优化版)

核心创新：信息不通过显式的注意力矩阵传播，而是像物理场一样在特征空间中传播。
每个 token 产生一个"场"，其他 token 感知这个场的梯度来获取信息。

时间复杂度: O(n * k) 其中 k 是场传播的迭代次数（通常 k << n）
空间复杂度: O(n * d) 其中 d 是维度

优化:
- 使用 torch.compile 加速
- 向量化传感器计算
- 减少中间张量分配
- 融合操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# 检查是否支持 torch.compile
_USE_COMPILE = hasattr(torch, 'compile') and torch.cuda.is_available()

# 使用 torch.jit.script 加速热点函数
@torch.jit.script
def fused_weighted_sum(field_sources: torch.Tensor, intensities: torch.Tensor) -> torch.Tensor:
    """融合加权求和操作"""
    return (field_sources * intensities.unsqueeze(-1)).sum(dim=2)

@torch.jit.script
def fused_gradient_sigmoid(gradient: torch.Tensor, shifted_field: torch.Tensor) -> torch.Tensor:
    """融合梯度和 sigmoid 操作"""
    return gradient * torch.sigmoid(shifted_field)

@torch.jit.script
def compute_causal_gradients(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """向量化计算因果梯度 - 一阶和二阶差分"""
    batch, seq_len, dim = field.shape
    
    # 一阶后向差分: f(t) - f(t-1)
    grad1 = torch.zeros_like(field)
    grad1[:, 1:, :] = field[:, 1:, :] - field[:, :-1, :]
    
    # 二阶后向差分: f(t) - 2*f(t-1) + f(t-2)
    grad2 = torch.zeros_like(field)
    grad2[:, 2:, :] = field[:, 2:, :] - 2*field[:, 1:-1, :] + field[:, :-2, :]
    
    return grad1, grad2


# FieldGenerator 已废弃，功能合并到 FieldEffectLayer


class FieldPropagator(nn.Module):
    """
    场传播器（优化版 v2）：单次传播 + 融合操作
    
    优化：
    - 固定单次迭代
    - 移除循环
    - 预计算所有可预计算的值
    """
    
    def __init__(
        self,
        field_dim: int,
        num_iterations: int = 1,  # 忽略，固定为1
        diffusion_rate: float = 0.1,
        use_learned_kernel: bool = True
    ):
        super().__init__()
        self.field_dim = field_dim
        self.diffusion_rate = diffusion_rate
        self.one_minus_diffusion = 1 - diffusion_rate
        
        # 使用 2-tap 核进一步加速
        self.propagation_kernel = nn.Parameter(torch.zeros(field_dim, 1, 2))
        nn.init.normal_(self.propagation_kernel, std=0.02)
        
        # 融合 decay 和 diffusion
        self.combined_weight = nn.Parameter(torch.ones(field_dim) * 0.09)  # diffusion * decay
        
        # 非线性场交互
        self.field_interaction = nn.Linear(field_dim, field_dim, bias=False)
    
    def forward(
        self,
        field_sources: torch.Tensor,
        intensities: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """单次场传播（最大速度）"""
        batch, seq_len, num_components, field_dim = field_sources.shape
        
        # 初始化场
        field = fused_weighted_sum(field_sources, intensities)
        
        # 单次因果卷积（2-tap: [t-1, t]）
        field_t = field.transpose(1, 2).contiguous()  # (batch, dim, seq)
        kernel = F.softmax(self.propagation_kernel, dim=-1)
        propagated = F.conv1d(F.pad(field_t, (1, 0)), kernel, groups=field_dim)
        propagated = propagated.transpose(1, 2)
        
        # 融合操作
        field = field + propagated * self.combined_weight
        
        # 非线性交互
        field = field + self.field_interaction(field)
        
        if mask is not None:
            field = field * mask.unsqueeze(-1)
        
        return field


class FieldSensor(nn.Module):
    """
    场感知器（极速版）：最小化投影
    
    优化：
    - 移除梯度计算
    - 单一投影层
    """
    
    def __init__(self, field_dim: int, out_dim: int, num_sensors: int = 2):
        super().__init__()
        # 单一投影
        self.proj = nn.Linear(field_dim, out_dim, bias=False)
    
    def forward(self, field: torch.Tensor, original_x: torch.Tensor) -> torch.Tensor:
        """直接投影"""
        return self.proj(field)


class FieldEffectLayer(nn.Module):
    """
    场效应层（极速一体化版）：融合所有组件
    
    极限优化：
    - 融合 Generator + Propagator + Sensor
    - 最小化层数和内存分配
    - 单一因果卷积实现场效应
    """
    
    def __init__(
        self,
        dim: int,
        field_dim: Optional[int] = None,
        num_iterations: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        
        # 输入归一化
        self.norm = nn.LayerNorm(dim)
        
        # 融合：投影 + 因果卷积 + 输出投影
        # 使用单一因果卷积实现"场传播"效果
        self.causal_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=0, groups=min(dim, 8), bias=False)
        
        # 非线性变换
        self.transform = nn.Linear(dim, dim, bias=False)
        
        # 可学习混合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        极简前向传播
        """
        residual = x
        x = self.norm(x)
        
        # 因果卷积（场传播效果）
        x_t = x.transpose(1, 2)  # (B, D, L)
        x_t = F.pad(x_t, (2, 0))  # 左填充保证因果性
        conv_out = self.causal_conv(x_t).transpose(1, 2)  # (B, L, D)
        
        # 非线性变换
        transformed = self.transform(x)
        
        # 混合
        output = self.alpha * conv_out + (1 - self.alpha) * transformed
        output = self.dropout(output)
        
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        
        return residual + output
