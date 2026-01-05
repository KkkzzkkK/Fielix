"""
Nexus Architecture - Dynamic Topology Module
动态拓扑连接模块

核心创新：不使用固定的全连接注意力模式，而是根据输入内容动态生成稀疏连接图
- 每个 token 动态决定要连接哪些其他 token
- 连接是稀疏的，降低计算复杂度
- 连接模式是可学习的，能自动发现重要的依赖关系

复杂度：O(n * k) 其中 k 是每个 token 的平均连接数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class TopologyPredictor(nn.Module):
    """
    拓扑预测器：预测每个 token 应该连接哪些其他 token
    
    使用两阶段方法：
    1. 粗粒度：快速筛选候选连接
    2. 细粒度：精确选择最终连接
    """
    
    def __init__(
        self,
        dim: int,
        max_connections: int = 32,
        temperature: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_connections = max_connections
        self.temperature = temperature
        
        # 连接意图编码器（编码"我想连接什么类型的 token"）
        self.intent_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        
        # 连接特征编码器（编码"我能提供什么"）
        self.feature_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        
        # 局部性偏置生成器（生成位置相关的连接偏置）
        self.locality_bias = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.locality_bias, std=0.1)
        
        # 连接数预测器（预测每个 token 需要多少连接）
        self.num_connections_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
    
    def _compute_locality_scores(
        self, 
        seq_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        计算局部性分数（近的 token 得分更高）
        
        Args:
            seq_len: 序列长度
            device: 设备
        
        Returns:
            locality_scores: (seq_len, seq_len)
        """
        # 创建相对位置矩阵
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
        
        # 使用可学习的局部性偏置
        # 将相对位置映射到偏置索引
        bias_len = self.locality_bias.shape[-1]
        rel_pos_clamped = rel_pos.clamp(-bias_len // 2, bias_len // 2 - 1) + bias_len // 2
        
        # 获取偏置值
        locality_scores = self.locality_bias[0, 0, rel_pos_clamped.long()]
        
        return locality_scores
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测连接拓扑
        
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) padding mask
            causal: 是否使用因果掩码
        
        Returns:
            topology: (batch, seq_len, max_connections) 连接索引
            weights: (batch, seq_len, max_connections) 连接权重
        """
        batch, seq_len, dim = x.shape
        device = x.device
        
        # 编码连接意图和特征
        intents = self.intent_encoder(x)  # (batch, seq_len, dim)
        features = self.feature_encoder(x)  # (batch, seq_len, dim)
        
        # 计算相似度分数
        scores = torch.matmul(intents, features.transpose(-2, -1))  # (batch, seq_len, seq_len)
        scores = scores / math.sqrt(dim)
        
        # 添加局部性偏置
        locality = self._compute_locality_scores(seq_len, device)
        scores = scores + locality.unsqueeze(0)
        
        # 应用因果掩码
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
        
        # 应用 padding 掩码
        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
            scores = scores.masked_fill(~mask_2d, float('-inf'))
        
        # 预测每个 token 的连接数
        num_conn_ratio = self.num_connections_predictor(x).squeeze(-1)  # (batch, seq_len)
        adaptive_k = (num_conn_ratio * self.max_connections).clamp(min=1).long()
        
        # 使用 top-k 选择连接（这里使用固定 k 简化，实际可以使用自适应 k）
        k = min(self.max_connections, seq_len)
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        
        # 计算连接权重（softmax over selected connections）
        weights = F.softmax(top_scores / self.temperature, dim=-1)
        
        return top_indices, weights


class SparseMessagePassing(nn.Module):
    """
    稀疏消息传递：在动态拓扑上进行信息交换
    
    每个 token 只与其连接的 token 交换信息
    """
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 消息生成
        self.message_proj = nn.Linear(dim, dim)
        
        # 消息聚合
        self.aggregate_proj = nn.Linear(dim, dim)
        
        # 更新门
        self.update_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(
        self,
        x: torch.Tensor,
        topology: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        在动态拓扑上传递消息
        
        Args:
            x: (batch, seq_len, dim)
            topology: (batch, seq_len, k) 连接索引
            weights: (batch, seq_len, k) 连接权重
        
        Returns:
            output: (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape
        _, _, k = topology.shape
        
        # 生成消息
        messages = self.message_proj(x)  # (batch, seq_len, dim)
        
        # 收集连接的 token 的消息
        # 使用 gather 操作
        topology_expanded = topology.unsqueeze(-1).expand(-1, -1, -1, dim)
        messages_expanded = messages.unsqueeze(2).expand(-1, -1, k, -1)
        
        # 高效的 gather
        gathered_messages = torch.gather(
            messages.unsqueeze(1).expand(-1, seq_len, -1, -1),
            dim=2,
            index=topology_expanded
        )  # (batch, seq_len, k, dim)
        
        # 加权聚合
        weighted_messages = gathered_messages * weights.unsqueeze(-1)
        aggregated = weighted_messages.sum(dim=2)  # (batch, seq_len, dim)
        
        # 聚合投影
        aggregated = self.aggregate_proj(aggregated)
        
        # 门控更新
        gate_input = torch.cat([x, aggregated], dim=-1)
        gate = self.update_gate(gate_input)
        
        # 更新表示
        updated = x * (1 - gate) + aggregated * gate
        
        # 输出投影
        output = self.output_proj(updated)
        
        return output


class HierarchicalTopology(nn.Module):
    """
    层次化拓扑：在多个尺度上建立连接
    
    - 局部连接：相邻 token 之间
    - 区块连接：token 块之间
    - 全局连接：关键 token 之间
    """
    
    def __init__(
        self,
        dim: int,
        num_levels: int = 3,
        block_sizes: List[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        
        # 默认块大小
        if block_sizes is None:
            block_sizes = [1, 4, 16][:num_levels]
        self.block_sizes = block_sizes
        
        # 每个层级的拓扑预测器
        self.level_predictors = nn.ModuleList([
            TopologyPredictor(dim, max_connections=32 // (2 ** i))
            for i in range(num_levels)
        ])
        
        # 每个层级的消息传递
        self.level_message_passing = nn.ModuleList([
            SparseMessagePassing(dim)
            for _ in range(num_levels)
        ])
        
        # 块聚合器（将 token 聚合为块表示）
        self.block_aggregators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU()
            )
            for _ in range(num_levels - 1)
        ])
        
        # 块分发器（将块信息分发回 token）
        self.block_distributors = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in range(num_levels - 1)
        ])
        
        # 层级融合
        self.level_fusion = nn.Linear(dim * num_levels, dim)
    
    def _aggregate_to_blocks(
        self, 
        x: torch.Tensor, 
        block_size: int,
        aggregator: nn.Module
    ) -> torch.Tensor:
        """
        将 token 聚合为块
        
        Args:
            x: (batch, seq_len, dim)
            block_size: 块大小
            aggregator: 聚合网络
        
        Returns:
            blocks: (batch, num_blocks, dim)
        """
        batch, seq_len, dim = x.shape
        
        # 填充到能被 block_size 整除
        pad_len = (block_size - seq_len % block_size) % block_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # 重塑为块
        num_blocks = (seq_len + pad_len) // block_size
        blocks = x.view(batch, num_blocks, block_size, dim)
        
        # 聚合（使用平均 + 聚合器）
        block_repr = blocks.mean(dim=2)
        block_repr = aggregator(block_repr)
        
        return block_repr, num_blocks
    
    def _distribute_from_blocks(
        self,
        block_repr: torch.Tensor,
        seq_len: int,
        block_size: int,
        distributor: nn.Module
    ) -> torch.Tensor:
        """
        将块信息分发回 token
        
        Args:
            block_repr: (batch, num_blocks, dim)
            seq_len: 原始序列长度
            block_size: 块大小
            distributor: 分发网络
        
        Returns:
            distributed: (batch, seq_len, dim)
        """
        batch, num_blocks, dim = block_repr.shape
        
        # 分发
        distributed = distributor(block_repr)
        
        # 扩展回 token 级别
        distributed = distributed.unsqueeze(2).expand(-1, -1, block_size, -1)
        distributed = distributed.reshape(batch, -1, dim)
        
        # 截断到原始长度
        distributed = distributed[:, :seq_len, :]
        
        return distributed
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> torch.Tensor:
        """
        层次化拓扑处理
        
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len)
            causal: 是否因果
        
        Returns:
            output: (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape
        level_outputs = []
        
        for level in range(self.num_levels):
            block_size = self.block_sizes[level]
            
            if block_size == 1:
                # Token 级别处理
                current_repr = x
            else:
                # 聚合到块级别
                current_repr, num_blocks = self._aggregate_to_blocks(
                    x, block_size, self.block_aggregators[level - 1]
                )
            
            # 预测拓扑
            topology, weights = self.level_predictors[level](
                current_repr, 
                causal=causal
            )
            
            # 消息传递
            processed = self.level_message_passing[level](
                current_repr, topology, weights
            )
            
            if block_size == 1:
                level_output = processed
            else:
                # 分发回 token 级别
                level_output = self._distribute_from_blocks(
                    processed, seq_len, block_size,
                    self.block_distributors[level - 1]
                )
            
            level_outputs.append(level_output)
        
        # 融合所有层级
        combined = torch.cat(level_outputs, dim=-1)
        output = self.level_fusion(combined)
        
        return output


class DynamicTopologyLayer(nn.Module):
    """
    动态拓扑层：Nexus 架构的核心连接层
    
    集成了：
    - 动态拓扑预测
    - 稀疏消息传递
    - 层次化连接
    """
    
    def __init__(
        self,
        dim: int,
        use_hierarchical: bool = True,
        num_levels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.use_hierarchical = use_hierarchical
        
        # 输入归一化
        self.norm = nn.LayerNorm(dim)
        
        if use_hierarchical:
            self.topology_module = HierarchicalTopology(dim, num_levels)
        else:
            self.topology_predictor = TopologyPredictor(dim)
            self.message_passing = SparseMessagePassing(dim)
        
        # 输出投影
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len)
            causal: 是否因果
        
        Returns:
            output: (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        
        if self.use_hierarchical:
            processed = self.topology_module(x, mask, causal)
        else:
            topology, weights = self.topology_predictor(x, mask, causal)
            processed = self.message_passing(x, topology, weights)
        
        output = self.output_proj(processed)
        output = self.dropout(output)
        
        return residual + output
