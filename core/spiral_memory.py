"""
Nexus Architecture - Spiral Memory Module
螺旋记忆模块

核心创新：信息在多个时间尺度上螺旋式积累，模拟人脑的记忆巩固过程
- 短期记忆：快速更新，快速遗忘
- 中期记忆：中等更新速度，选择性保留
- 长期记忆：缓慢更新，持久保留

螺旋结构：信息从短期逐渐"螺旋"到长期，在这个过程中被压缩和抽象化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class MemoryCell(nn.Module):
    """
    单个记忆单元，具有可学习的更新和遗忘门控
    
    与 LSTM 不同，这里使用"软性压缩"而非硬门控
    """
    
    def __init__(self, dim: int, compression_ratio: float = 0.5):
        super().__init__()
        self.dim = dim
        self.compression_ratio = compression_ratio
        
        # 记忆更新门（决定多少新信息进入记忆）
        self.update_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 记忆保持门（决定多少旧信息保留）
        self.retain_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 记忆压缩器（提取信息的精华）
        compressed_dim = int(dim * compression_ratio)
        self.compressor = nn.Sequential(
            nn.Linear(dim, compressed_dim),
            nn.GELU(),
            nn.Linear(compressed_dim, dim)
        )
        
        # 记忆融合器
        self.fusion = nn.Linear(dim * 2, dim)
    
    def forward(
        self, 
        memory: torch.Tensor, 
        new_info: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新记忆单元
        
        Args:
            memory: (batch, dim) 当前记忆状态
            new_info: (batch, dim) 新输入信息
        
        Returns:
            updated_memory: (batch, dim) 更新后的记忆
            overflow: (batch, dim) 溢出的信息（传递给下一层）
        """
        # 拼接记忆和新信息
        combined = torch.cat([memory, new_info], dim=-1)
        
        # 计算门控值
        update_weight = self.update_gate(combined)
        retain_weight = self.retain_gate(combined)
        
        # 压缩新信息
        compressed_new = self.compressor(new_info)
        
        # 更新记忆
        retained_memory = memory * retain_weight
        new_memory = compressed_new * update_weight
        updated_memory = retained_memory + new_memory
        
        # 计算溢出（被"挤出"的信息，用于传递给更长期的记忆）
        overflow_raw = torch.cat([
            memory * (1 - retain_weight),  # 被遗忘的记忆
            new_info * (1 - update_weight)  # 未被吸收的新信息
        ], dim=-1)
        overflow = self.fusion(overflow_raw)
        
        return updated_memory, overflow


class SpiralMemoryBank(nn.Module):
    """
    螺旋记忆库：多层级记忆系统
    
    结构：
    Level 0 (短期): 快速更新，容量小
    Level 1 (中期): 中速更新，容量中等
    Level 2 (长期): 慢速更新，容量大
    ...
    
    信息流动：
    新信息 -> Level 0 -> 溢出 -> Level 1 -> 溢出 -> Level 2 -> ...
    """
    
    def __init__(
        self,
        dim: int,
        num_levels: int = 3,
        slots_per_level: List[int] = None,
        update_rates: List[float] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        
        # 默认配置：每层级记忆槽数量递增
        if slots_per_level is None:
            slots_per_level = [4, 8, 16][:num_levels]
        self.slots_per_level = slots_per_level
        
        # 默认配置：更新率递减
        if update_rates is None:
            update_rates = [0.9, 0.5, 0.1][:num_levels]
        self.update_rates = update_rates
        
        # 为每个层级创建记忆单元
        self.memory_cells = nn.ModuleList([
            MemoryCell(dim, compression_ratio=0.5 + 0.1 * i)
            for i in range(num_levels)
        ])
        
        # 记忆槽选择器（决定更新哪个槽）
        self.slot_selectors = nn.ModuleList([
            nn.Linear(dim, slots)
            for slots in slots_per_level
        ])
        
        # 记忆读取器（使用场效应方式替代 MultiheadAttention，保持风格统一）
        self.memory_query_proj = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_levels)
        ])
        self.memory_key_proj = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_levels)
        ])
        self.memory_value_proj = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_levels)
        ])
        self.memory_out_proj = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_levels)
        ])
        
        # 层级间转换
        self.level_transforms = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in range(num_levels - 1)
        ])
        
        # 输出融合
        self.output_fusion = nn.Linear(dim * num_levels, dim)
    
    def _select_slots(
        self, 
        info: torch.Tensor, 
        level: int, 
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        软性选择要更新的记忆槽
        
        Args:
            info: (batch, dim)
            level: 记忆层级
            temperature: 选择的温度参数
        
        Returns:
            selection_weights: (batch, num_slots)
        """
        logits = self.slot_selectors[level](info)
        weights = F.softmax(logits / temperature, dim=-1)
        return weights
    
    def write(
        self,
        memories: List[torch.Tensor],
        new_info: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        写入新信息到记忆系统（向量化版本，提升效率）
        
        Args:
            memories: 每个层级的记忆 [(batch, slots_i, dim), ...]
            new_info: (batch, dim) 新信息
        
        Returns:
            updated_memories: 更新后的记忆列表
        """
        batch_size = new_info.shape[0]
        updated_memories = []
        current_info = new_info
        
        for level in range(self.num_levels):
            memory = memories[level]  # (batch, slots, dim)
            num_slots = self.slots_per_level[level]
            update_rate = self.update_rates[level]
            
            # 选择要更新的槽
            slot_weights = self._select_slots(current_info, level)  # (batch, slots)
            
            # 向量化更新所有槽
            # 扩展 current_info 到所有槽: (batch, 1, dim) -> (batch, slots, dim)
            info_expanded = current_info.unsqueeze(1).expand(-1, num_slots, -1)
            weights_expanded = slot_weights.unsqueeze(-1)  # (batch, slots, 1)
            
            # 计算每个槽的加权新信息
            slot_new_info = info_expanded * weights_expanded * update_rate  # (batch, slots, dim)
            
            # 向量化门控计算
            combined = torch.cat([memory, slot_new_info], dim=-1)  # (batch, slots, dim*2)
            combined_flat = combined.view(batch_size * num_slots, -1)
            
            # 使用记忆单元的门控（向量化）
            cell = self.memory_cells[level]
            update_weight = cell.update_gate(combined_flat).view(batch_size, num_slots, -1)
            retain_weight = cell.retain_gate(combined_flat).view(batch_size, num_slots, -1)
            
            # 压缩新信息
            compressed_new = cell.compressor(slot_new_info.view(-1, self.dim)).view(batch_size, num_slots, -1)
            
            # 更新记忆
            retained_memory = memory * retain_weight
            new_memory = compressed_new * update_weight
            updated_memory = retained_memory + new_memory
            
            # 计算溢出（向量化）
            overflow_raw = torch.cat([
                memory * (1 - retain_weight),
                slot_new_info * (1 - update_weight)
            ], dim=-1)
            overflow = cell.fusion(overflow_raw.view(-1, self.dim * 2)).view(batch_size, num_slots, -1)
            level_overflow = (overflow * weights_expanded).sum(dim=1)  # (batch, dim)
            
            updated_memories.append(updated_memory)
            
            # 准备下一层的输入
            if level < self.num_levels - 1:
                current_info = self.level_transforms[level](level_overflow)
        
        return updated_memories
    
    def read(
        self,
        memories: List[torch.Tensor],
        query: torch.Tensor
    ) -> torch.Tensor:
        """
        从记忆系统读取相关信息（场效应风格，保持架构统一）
        
        Args:
            memories: 每个层级的记忆 [(batch, slots_i, dim), ...]
            query: (batch, seq_len, dim) 查询
        
        Returns:
            retrieved: (batch, seq_len, dim) 检索到的信息
        """
        level_outputs = []
        batch, seq_len, dim = query.shape
        
        for level in range(self.num_levels):
            memory = memories[level]  # (batch, slots, dim)
            num_slots = memory.shape[1]
            
            # 场效应风格的记忆读取
            q = self.memory_query_proj[level](query)  # (batch, seq_len, dim)
            k = self.memory_key_proj[level](memory)   # (batch, slots, dim)
            v = self.memory_value_proj[level](memory) # (batch, slots, dim)
            
            # 计算相似度分数（缩放点积）
            scale = dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch, seq_len, slots)
            
            # 数值稳定的 softmax
            scores = scores - scores.max(dim=-1, keepdim=True).values
            attn = F.softmax(scores, dim=-1)
            
            # 加权求和
            retrieved = torch.matmul(attn, v)  # (batch, seq_len, dim)
            retrieved = self.memory_out_proj[level](retrieved)
            level_outputs.append(retrieved)
        
        # 融合所有层级的信息
        combined = torch.cat(level_outputs, dim=-1)
        output = self.output_fusion(combined)
        
        return output
    
    def init_memory(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """
        初始化记忆状态
        
        Args:
            batch_size: 批次大小
            device: 设备
        
        Returns:
            memories: 初始化的记忆列表
        """
        memories = []
        for num_slots in self.slots_per_level:
            memory = torch.zeros(batch_size, num_slots, self.dim, device=device)
            memories.append(memory)
        return memories


class SpiralMemoryLayer(nn.Module):
    """
    螺旋记忆层：将螺旋记忆集成到 Nexus 架构中
    
    这一层在处理序列时维护和更新记忆状态
    """
    
    def __init__(
        self,
        dim: int,
        num_levels: int = 3,
        write_every_n: int = 4,  # 每处理 n 个 token 写入一次
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.write_every_n = write_every_n
        
        # 记忆库
        self.memory_bank = SpiralMemoryBank(dim, num_levels)
        
        # 写入信息压缩器（将 n 个 token 压缩为一个记忆条目）
        self.write_compressor = nn.Sequential(
            nn.Linear(dim * write_every_n, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        # 归一化
        self.norm = nn.LayerNorm(dim)
        
        # 门控（决定记忆信息的影响程度）
        self.memory_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memories: Optional[List[torch.Tensor]] = None,
        return_memories: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        处理序列并更新记忆
        
        Args:
            x: (batch, seq_len, dim) 输入序列
            memories: 可选的初始记忆状态
            return_memories: 是否返回更新后的记忆
        
        Returns:
            output: (batch, seq_len, dim) 输出序列
            memories: 可选的更新后记忆
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        
        # 初始化记忆（如果需要）
        if memories is None:
            memories = self.memory_bank.init_memory(batch_size, device)
        
        residual = x
        x = self.norm(x)
        
        # 1. 从记忆中读取相关信息
        memory_info = self.memory_bank.read(memories, x)
        
        # 2. 计算记忆门控
        gate_input = torch.cat([x, memory_info], dim=-1)
        gate = self.memory_gate(gate_input)
        
        # 3. 融合记忆信息
        enhanced = x + gate * memory_info
        
        # 4. 写入新记忆（每 write_every_n 个 token）
        num_writes = seq_len // self.write_every_n
        if num_writes > 0:
            # 将序列分块
            truncated_len = num_writes * self.write_every_n
            chunks = x[:, :truncated_len, :].view(
                batch_size, num_writes, self.write_every_n, dim
            )
            
            # 压缩每个块
            chunks_flat = chunks.view(batch_size * num_writes, -1)
            write_info = self.write_compressor(chunks_flat)
            write_info = write_info.view(batch_size, num_writes, dim)
            
            # 逐个写入
            for i in range(num_writes):
                memories = self.memory_bank.write(memories, write_info[:, i, :])
        
        # 应用 dropout
        output = self.dropout(enhanced)
        
        # 残差连接
        output = residual + output
        
        if return_memories:
            return output, memories
        return output, None


class RecurrentSpiralMemory(nn.Module):
    """
    循环螺旋记忆：支持流式处理的螺旋记忆
    
    特别适合长序列和流式生成任务
    """
    
    def __init__(
        self,
        dim: int,
        num_levels: int = 4,
        chunk_size: int = 64
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.chunk_size = chunk_size
        
        # 主记忆层
        self.memory_layer = SpiralMemoryLayer(
            dim, 
            num_levels, 
            write_every_n=chunk_size // 4
        )
        
        # 块间状态转换
        self.state_transform = nn.GRUCell(dim, dim)
        
        # 块总结器
        self.chunk_summarizer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[List[torch.Tensor], torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], torch.Tensor]]:
        """
        流式处理序列
        
        Args:
            x: (batch, seq_len, dim) 输入块
            state: (memories, hidden) 前一状态
        
        Returns:
            output: (batch, seq_len, dim)
            new_state: 更新后的状态
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        
        # 解包状态
        if state is None:
            memories = None
            hidden = torch.zeros(batch_size, dim, device=device)
        else:
            memories, hidden = state
        
        # 处理当前块
        output, memories = self.memory_layer(x, memories, return_memories=True)
        
        # 计算块摘要
        chunk_summary = self.chunk_summarizer(output.mean(dim=1))
        
        # 更新隐藏状态
        new_hidden = self.state_transform(chunk_summary, hidden)
        
        # 将隐藏状态信息注入输出
        output = output + new_hidden.unsqueeze(1) * 0.1
        
        return output, (memories, new_hidden)
