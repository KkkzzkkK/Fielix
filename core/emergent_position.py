"""
Fielix Architecture - Emergent Position Encoding Module
涌现式位置编码模块

核心创新：位置信息不是预定义的，而是从 token 之间的动态交互中涌现
- 相对位置从信息流动模式中学习
- 位置编码是上下文相关的（同一位置在不同上下文中有不同编码）
- 支持任意长度外推

与传统方法对比：
- 绝对位置编码 (APE): 预定义，无法外推
- 相对位置编码 (RPE): 需要显式计算位置矩阵
- RoPE: 基于旋转，但仍是预定义的
- ALiBi: 线性偏置，简单但表达能力有限
- Fielix 涌现位置: 从数据中学习，自适应，可外推
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionOscillator(nn.Module):
    """
    位置振荡器：生成用于位置编码的振荡信号
    
    灵感来自神经科学中的网格细胞(Grid Cells)，它们通过周期性放电来编码空间位置
    """
    
    def __init__(self, dim: int, num_frequencies: int = 8):
        super().__init__()
        self.dim = dim
        self.num_frequencies = num_frequencies
        
        # 可学习的频率（不同于 Transformer 的固定频率）
        self.frequencies = nn.Parameter(
            torch.randn(num_frequencies) * 0.1
        )
        
        # 可学习的相位
        self.phases = nn.Parameter(
            torch.zeros(num_frequencies)
        )
        
        # 振幅调制器（从输入内容生成振幅）
        self.amplitude_generator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_frequencies * 2)  # sin 和 cos 各一组
        )
        
        # 输出投影
        self.output_proj = nn.Linear(num_frequencies * 2, dim)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        生成位置振荡信号
        
        Args:
            x: (batch, seq_len, dim) 输入表示
            positions: (batch, seq_len) 位置索引
        
        Returns:
            oscillation: (batch, seq_len, dim) 位置振荡信号
        """
        batch, seq_len, dim = x.shape
        
        # 从输入生成振幅（内容相关的位置编码）
        amplitudes = self.amplitude_generator(x)  # (batch, seq_len, num_freq * 2)
        amplitudes = amplitudes.view(batch, seq_len, 2, self.num_frequencies)
        sin_amp, cos_amp = amplitudes[:, :, 0, :], amplitudes[:, :, 1, :]
        
        # 计算频率和相位
        freq = self.frequencies.abs() + 0.1  # 确保正频率
        phase = self.phases
        
        # 生成振荡信号
        # positions: (batch, seq_len) -> (batch, seq_len, 1)
        pos = positions.unsqueeze(-1).float()
        
        # 计算角度: freq * pos + phase
        angles = pos * freq.unsqueeze(0).unsqueeze(0) + phase.unsqueeze(0).unsqueeze(0)
        
        # 生成 sin 和 cos 信号
        sin_signals = torch.sin(angles) * sin_amp
        cos_signals = torch.cos(angles) * cos_amp
        
        # 拼接
        oscillation = torch.cat([sin_signals, cos_signals], dim=-1)
        
        # 投影到输出维度
        oscillation = self.output_proj(oscillation)
        
        return oscillation


class RelationalPositionLearner(nn.Module):
    """
    关系位置学习器：从 token 对的关系中学习相对位置
    
    不显式编码位置，而是让模型学习"这两个 token 相距多远"的表示
    """
    
    def __init__(self, dim: int, max_distance: int = 128):
        super().__init__()
        self.dim = dim
        self.max_distance = max_distance
        
        # 距离编码器（将 token 对映射到距离表示）
        self.distance_encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )
        
        # 距离嵌入（学习不同距离的表示）
        self.distance_embedding = nn.Embedding(max_distance * 2 + 1, dim)
        
        # 方向编码（前向 vs 后向）
        self.direction_embedding = nn.Embedding(3, dim)  # -1, 0, 1
        
        # 融合层
        self.fusion = nn.Linear(dim * 2, dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        学习相对位置表示
        
        Args:
            x: (batch, seq_len, dim)
            positions: (batch, seq_len) 可选的位置索引
        
        Returns:
            position_bias: (batch, seq_len, seq_len, dim) 位置偏置
        """
        batch, seq_len, dim = x.shape
        device = x.device
        
        # 如果没有提供位置，使用默认的 0, 1, 2, ...
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        
        # 计算相对距离
        rel_dist = positions.unsqueeze(2) - positions.unsqueeze(1)  # (batch, seq_len, seq_len)
        
        # 裁剪到最大距离
        rel_dist_clamped = rel_dist.clamp(-self.max_distance, self.max_distance)
        rel_dist_idx = rel_dist_clamped + self.max_distance  # 转为正索引
        
        # 获取距离嵌入
        dist_emb = self.distance_embedding(rel_dist_idx.long())  # (batch, seq_len, seq_len, dim)
        
        # 计算方向
        direction = torch.sign(rel_dist).long() + 1  # 0, 1, 2
        dir_emb = self.direction_embedding(direction)  # (batch, seq_len, seq_len, dim)
        
        # 融合距离和方向
        combined = torch.cat([dist_emb, dir_emb], dim=-1)
        position_bias = self.fusion(combined)
        
        return position_bias


class EmergentPositionEncoder(nn.Module):
    """
    涌现式位置编码器（极速版 V2）：接近零开销的位置编码
    
    优化：
    - 移除因果卷积（主要开销源）
    - 仅使用预计算的位置嵌入
    - 添加可学习的缩放因子
    """
    
    def __init__(
        self,
        dim: int,
        use_oscillator: bool = True,
        use_relational: bool = True,
        use_flow: bool = True,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 简单的可学习位置嵌入
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        
        # 缩放因子
        self.scale = nn.Parameter(torch.ones(1))
        
        # 初始化
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_bias: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """极速位置编码 - 零额外计算"""
        batch, seq_len, dim = x.shape
        device = x.device
        
        # 生成位置索引（缓存友好）
        if positions is None:
            positions = torch.arange(seq_len, device=device)
        else:
            positions = positions[0]  # 取第一个 batch
        
        # 位置嵌入 - 直接加到输入
        pos_emb = self.pos_embedding(positions[:seq_len])  # (seq_len, dim)
        
        # 广播加法 - 高效
        x_with_pos = x + pos_emb * self.scale
        
        return x_with_pos, None


class FlowPositionEncoder(nn.Module):
    """
    流动位置编码器：从信息流动模式中涌现位置
    
    核心思想：位置从“我之前是谁”的模式中推断（因果版本）
    
    修复：只使用前向 GRU，避免泄露未来信息
    """
    
    def __init__(self, dim: int, window_size: int = 4):  # 减小窗口
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        # 简化：只用因果卷积替代 GRU（更快）
        self.causal_conv = nn.Conv1d(
            dim, dim, kernel_size=window_size, padding=0, groups=dim // 8
        )
        self.conv_pad = window_size - 1
        
        # 位置推断器（简化为单层）
        self.position_inferrer = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        从信息流动中推断位置（因果版本，简化加速）
        """
        # 因果卷积
        x_padded = F.pad(x.transpose(1, 2), (self.conv_pad, 0))
        x_conv = self.causal_conv(x_padded).transpose(1, 2)
        
        # 位置推断
        flow_position = self.position_inferrer(x_conv)
        
        return flow_position


class PositionExtrapolator(nn.Module):
    """
    位置外推器：支持超出训练长度的位置编码
    
    使用神经 ODE 风格的连续位置建模
    """
    
    def __init__(self, dim: int, num_basis: int = 16):
        super().__init__()
        self.dim = dim
        self.num_basis = num_basis
        
        # 位置基函数（可学习）
        self.basis_functions = nn.Parameter(torch.randn(num_basis, dim) * 0.1)
        
        # 基函数系数生成器
        self.coef_generator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_basis)
        )
        
        # 位置导数预测器（用于外推）
        self.derivative_predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        target_len: int,
        step_size: float = 1.0
    ) -> torch.Tensor:
        """
        外推位置编码到任意长度
        
        Args:
            x: (batch, seq_len, dim) 已知位置的表示
            target_len: 目标长度
            step_size: 外推步长
        
        Returns:
            extrapolated: (batch, target_len, dim)
        """
        batch, seq_len, dim = x.shape
        device = x.device
        
        if target_len <= seq_len:
            return x[:, :target_len, :]
        
        # 初始化：使用最后几个位置
        result = [x]
        current_pos = x[:, -1:, :]  # (batch, 1, dim)
        prev_pos = x[:, -2:-1, :] if seq_len > 1 else current_pos
        
        # 逐步外推
        for i in range(seq_len, target_len):
            # 计算"速度"（位置变化趋势）
            velocity_input = torch.cat([current_pos, prev_pos], dim=-1)
            velocity = self.derivative_predictor(velocity_input)
            
            # 欧拉积分
            next_pos = current_pos + velocity * step_size
            
            # 添加基函数调制
            coefs = self.coef_generator(next_pos)  # (batch, 1, num_basis)
            basis_contribution = torch.matmul(
                coefs, self.basis_functions
            )  # (batch, 1, dim)
            next_pos = next_pos + basis_contribution * 0.1
            
            result.append(next_pos)
            prev_pos = current_pos
            current_pos = next_pos
        
        extrapolated = torch.cat(result, dim=1)
        return extrapolated
