"""
Nexus Architecture - Field Effect Propagation Module
场效应传播模块

核心创新：信息不通过显式的注意力矩阵传播，而是像物理场一样在特征空间中传播。
每个 token 产生一个"场"，其他 token 感知这个场的梯度来获取信息。

时间复杂度: O(n * k) 其中 k 是场传播的迭代次数（通常 k << n）
空间复杂度: O(n * d) 其中 d 是维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# 使用 torch.jit.script 加速热点函数
@torch.jit.script
def fused_weighted_sum(field_sources: torch.Tensor, intensities: torch.Tensor) -> torch.Tensor:
    """融合加权求和操作"""
    return (field_sources * intensities.unsqueeze(-1)).sum(dim=2)

@torch.jit.script  
def fused_gradient_sigmoid(gradient: torch.Tensor, shifted_field: torch.Tensor) -> torch.Tensor:
    """融合梯度和 sigmoid 操作"""
    return gradient * torch.sigmoid(shifted_field)


class FieldGenerator(nn.Module):
    """
    场生成器：将 token 表示转换为场源
    
    每个 token 生成三种场分量：
    1. 信息场 (Information Field): 携带语义信息
    2. 吸引场 (Attraction Field): 决定哪些 token 应该被关注
    3. 时序场 (Temporal Field): 编码相对位置信息
    """
    
    def __init__(self, dim: int, field_dim: int, num_field_components: int = 3):
        super().__init__()
        self.dim = dim
        self.field_dim = field_dim
        self.num_components = num_field_components
        
        # 场源投影
        self.field_source_proj = nn.Linear(dim, field_dim * num_field_components)
        
        # 场强度调制器（简化版，加速）
        self.intensity_gate = nn.Linear(dim, num_field_components)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.field_source_proj.weight, gain=0.1)
        nn.init.zeros_(self.field_source_proj.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            field_sources: (batch, seq_len, num_components, field_dim)
            intensities: (batch, seq_len, num_components)
        """
        batch, seq_len, _ = x.shape
        
        # 生成场源
        field_sources = self.field_source_proj(x)
        field_sources = field_sources.view(batch, seq_len, self.num_components, self.field_dim)
        
        # 计算场强度（用 sigmoid 归一化）
        intensities = torch.sigmoid(self.intensity_gate(x))
        
        return field_sources, intensities


class FieldPropagator(nn.Module):
    """
    场传播器：使用迭代扩散模拟场的传播
    
    关键创新：使用可学习的"场方程"来控制信息如何在序列中传播
    类似于热传导方程，但参数是可学习的
    """
    
    def __init__(
        self, 
        field_dim: int, 
        num_iterations: int = 1,  # 单次传播，最快
        diffusion_rate: float = 0.1,
        use_learned_kernel: bool = True
    ):
        super().__init__()
        self.field_dim = field_dim
        self.num_iterations = num_iterations
        self.diffusion_rate = diffusion_rate
        self.use_learned_kernel = use_learned_kernel
        
        if use_learned_kernel:
            # 可学习的因果传播核（只向左传播，保证因果性）
            # 使用 [t-4, t-3, t-2, t-1, t] 的窗口
            self.propagation_kernel = nn.Parameter(torch.zeros(field_dim, 5))  # 5-tap causal kernel
            nn.init.normal_(self.propagation_kernel, std=0.02)
        
        # 因果模式标志
        self.causal = True
        
        # 场衰减因子（模拟场强随距离衰减）
        self.decay_factor = nn.Parameter(torch.ones(field_dim) * 0.9)
        
        # 非线性场交互（简化为单层）
        self.field_interaction = nn.Linear(field_dim, field_dim)
    
    def _apply_kernel(self, field: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        应用因果传播核（只从过去位置传播信息）
        Args:
            field: (batch, seq_len, field_dim)
            causal: 是否使用因果卷积
        Returns:
            propagated: (batch, seq_len, field_dim)
        """
        batch, seq_len, field_dim = field.shape
        
        # 转置以便卷积操作
        field_t = field.transpose(1, 2)  # (batch, field_dim, seq_len)
        
        # 构建深度可分离卷积核
        kernel = self.propagation_kernel.unsqueeze(1)  # (field_dim, 1, 5)
        kernel = F.softmax(kernel, dim=-1)  # 归一化
        
        if causal:
            # 因果卷积：左填充4，右填充0
            # 这样位置 t 只能看到 [t-4, t-3, t-2, t-1, t]
            field_t = F.pad(field_t, (4, 0))  # 左填充
            propagated = F.conv1d(field_t, kernel, padding=0, groups=field_dim)
        else:
            # 非因果：双侧填充
            propagated = F.conv1d(field_t, kernel, padding=2, groups=field_dim)
        
        return propagated.transpose(1, 2)
    
    def forward(
        self, 
        field_sources: torch.Tensor, 
        intensities: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        迭代传播场
        
        Args:
            field_sources: (batch, seq_len, num_components, field_dim)
            intensities: (batch, seq_len, num_components)
            mask: (batch, seq_len) - padding mask
        
        Returns:
            aggregated_field: (batch, seq_len, field_dim)
        """
        batch, seq_len, num_components, field_dim = field_sources.shape
        
        # 初始化场为加权场源（使用融合函数）
        field = fused_weighted_sum(field_sources, intensities)
        
        # 迭代传播
        for i in range(self.num_iterations):
            # 1. 应用传播核
            if self.use_learned_kernel:
                propagated = self._apply_kernel(field)
            else:
                # 简单的均值传播
                propagated = F.avg_pool1d(
                    field.transpose(1, 2), 
                    kernel_size=3, 
                    stride=1, 
                    padding=1
                ).transpose(1, 2)
            
            # 2. 场衰减（inplace 操作）
            propagated.mul_(self.decay_factor)
            
            # 3. 与原始场源混合（防止信息丢失）
            field = (1 - self.diffusion_rate) * field + self.diffusion_rate * propagated
            
            # 4. 非线性交互（场的自相互作用）
            if i == self.num_iterations - 1:
                field = field + self.field_interaction(field)
            
            # 5. 应用 mask
            if mask is not None:
                field = field * mask.unsqueeze(-1)
        
        return field


class FieldSensor(nn.Module):
    """
    场感知器：感知传播后的场，并转换回 token 表示
    
    创新点：使用"场梯度"来感知周围的信息，而不是直接查询
    """
    
    def __init__(self, field_dim: int, out_dim: int, num_sensors: int = 2):  # 减少传感器数量
        super().__init__()
        self.field_dim = field_dim
        self.out_dim = out_dim
        self.num_sensors = num_sensors
        
        # 多方向传感器（感知场在不同方向的梯度）
        self.sensors = nn.ModuleList([
            nn.Linear(field_dim, out_dim // num_sensors)
            for _ in range(num_sensors)
        ])
        
        # 梯度计算的可学习偏移
        self.gradient_offsets = nn.Parameter(torch.randn(num_sensors, field_dim) * 0.1)
        
        # 输出融合
        self.output_fusion = nn.Linear(out_dim, out_dim)
    
    def _compute_field_gradient(
        self, 
        field: torch.Tensor, 
        sensor_idx: int
    ) -> torch.Tensor:
        """
        计算场在特定方向的梯度
        
        Args:
            field: (batch, seq_len, field_dim)
            sensor_idx: 传感器索引
        
        Returns:
            gradient: (batch, seq_len, field_dim)
        """
        batch, seq_len, field_dim = field.shape
        
        # 获取当前传感器的偏移
        offset = self.gradient_offsets[sensor_idx]  # (field_dim,)
        
        # 计算"位移"后的场（通过在特征空间中偏移）
        shifted_field = field + offset.unsqueeze(0).unsqueeze(0)
        
        # 时序梯度：全部使用因果（后向）差分，避免泄露未来信息
        gradient = torch.zeros_like(field)
        if sensor_idx % 2 == 0:
            # 一阶后向差分: f(t) - f(t-1)
            gradient[:, 1:, :] = field[:, 1:, :] - field[:, :-1, :]
        else:
            # 二阶后向差分: f(t) - 2*f(t-1) + f(t-2)
            gradient[:, 2:, :] = field[:, 2:, :] - 2*field[:, 1:-1, :] + field[:, :-2, :]
        
        # 结合特征空间的偏移（使用融合函数）
        return fused_gradient_sigmoid(gradient, shifted_field)
    
    def forward(self, field: torch.Tensor, original_x: torch.Tensor) -> torch.Tensor:
        """
        感知场并生成输出
        
        Args:
            field: (batch, seq_len, field_dim)
            original_x: (batch, seq_len, dim) - 原始输入，用于残差连接
        
        Returns:
            output: (batch, seq_len, out_dim)
        """
        sensor_outputs = []
        
        for i, sensor in enumerate(self.sensors):
            # 计算场梯度
            gradient = self._compute_field_gradient(field, i)
            
            # 结合场值和梯度
            sensed = field + 0.1 * gradient
            
            # 通过传感器
            sensor_out = sensor(sensed)
            sensor_outputs.append(sensor_out)
        
        # 拼接所有传感器输出
        combined = torch.cat(sensor_outputs, dim=-1)
        
        # 融合输出
        output = self.output_fusion(combined)
        
        return output


class FieldEffectLayer(nn.Module):
    """
    场效应层：完整的场效应传播层
    
    这是 Nexus 架构的核心组件，替代传统的自注意力层
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
        self.field_dim = field_dim or dim
        
        # 输入归一化
        self.norm = nn.LayerNorm(dim)
        
        # 场生成
        self.field_generator = FieldGenerator(dim, self.field_dim)
        
        # 场传播
        self.field_propagator = FieldPropagator(
            self.field_dim, 
            num_iterations=num_iterations
        )
        
        # 场感知
        self.field_sensor = FieldSensor(self.field_dim, dim)
        
        # 输出投影
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len)
        
        Returns:
            output: (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        
        # 1. 生成场源
        field_sources, intensities = self.field_generator(x)
        
        # 2. 传播场
        propagated_field = self.field_propagator(field_sources, intensities, mask)
        
        # 3. 感知场
        sensed = self.field_sensor(propagated_field, x)
        
        # 4. 输出投影
        output = self.output_proj(sensed)
        output = self.dropout(output)
        
        # 残差连接
        return residual + output
