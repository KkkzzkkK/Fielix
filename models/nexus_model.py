"""
Fielix Architecture - Complete Model Implementation
Fielix 架构完整模型实现

这是一个全新的神经网络架构，不同于 Transformer、RNN、SSM 等现有架构。

核心创新：
1. 场效应传播：信息像物理场一样在特征空间中传播
2. 动态拓扑：连接模式根据内容动态生成
3. 螺旋记忆：多时间尺度的记忆系统
4. 涌现位置：位置信息从交互中涌现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import sys
sys.path.append('..')
from core import (
    FielixBlock,
    EmergentPositionEncoder,
    FieldEffectLayer,
    DynamicTopologyLayer,
    SpiralMemoryLayer,
    FielixFeedForward
)


@dataclass
class FielixConfig:
    """Fielix 模型配置"""
    # 基本参数
    vocab_size: int = 32000
    dim: int = 768
    num_layers: int = 12
    max_seq_len: int = 8192
    
    # 注意力配置
    attention_type: str = 'hybrid'  # 'field', 'topology', 'hybrid'
    field_iterations: int = 4
    topology_levels: int = 3
    
    # 记忆配置
    use_memory: bool = True
    memory_levels: int = 3
    
    # 前馈配置
    ffn_type: str = 'gated'  # 'adaptive', 'moe', 'gated'
    ffn_dim: int = None  # None 表示自动计算
    num_experts: int = 8
    
    # 位置编码配置
    use_emergent_position: bool = True
    
    # 正则化
    dropout: float = 0.1
    
    # 其他
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = int(self.dim * 8 / 3)  # SwiGLU 最优比例


class FielixEmbedding(nn.Module):
    """
    Fielix 嵌入层
    
    特点：
    - 可学习的 token 嵌入
    - 可选的涌现式位置编码
    - 嵌入缩放
    """
    
    def __init__(self, config: FielixConfig):
        super().__init__()
        self.config = config
        
        # Token 嵌入
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.dim, 
            padding_idx=config.pad_token_id
        )
        
        # 涌现式位置编码
        if config.use_emergent_position:
            self.position_encoder = EmergentPositionEncoder(
                config.dim,
                use_oscillator=True,
                use_relational=False,  # 在嵌入层不使用关系编码
                use_flow=True,
                max_seq_len=config.max_seq_len
            )
        else:
            # 备选：简单的可学习位置嵌入
            self.position_embedding = nn.Embedding(config.max_seq_len, config.dim)
        
        # 嵌入缩放
        self.embed_scale = math.sqrt(config.dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if hasattr(self, 'position_embedding'):
            nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            positions: (batch, seq_len) 可选的位置索引
        
        Returns:
            embeddings: (batch, seq_len, dim)
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token 嵌入
        x = self.token_embedding(input_ids) * self.embed_scale
        
        # 位置编码
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        
        if self.config.use_emergent_position:
            x, _ = self.position_encoder(x, positions)
        else:
            pos_emb = self.position_embedding(positions)
            x = x + pos_emb
        
        x = self.dropout(x)
        
        return x


class FielixDecoder(nn.Module):
    """
    Fielix 解码器：堆叠多个 Fielix 块
    """
    
    def __init__(self, config: FielixConfig):
        super().__init__()
        self.config = config
        
        # 创建层
        self.layers = nn.ModuleList([
            FielixBlock(
                dim=config.dim,
                attention_type=config.attention_type,
                use_memory=(config.use_memory and i % 3 == 2),  # 每 3 层加一次记忆
                ffn_type=config.ffn_type,
                num_experts=config.num_experts,
                field_iterations=config.field_iterations,
                topology_levels=config.topology_levels,
                memory_levels=config.memory_levels,
                dropout=config.dropout
            )
            for i in range(config.num_layers)
        ])
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(config.dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        memory_states: Optional[List[List[torch.Tensor]]] = None,
        causal: bool = True
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]], torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len)
            memory_states: 每层的记忆状态
            causal: 是否因果
        
        Returns:
            output: (batch, seq_len, dim)
            new_memory_states: 更新后的记忆状态
            total_aux_loss: 总辅助损失
        """
        new_memory_states = []
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            # 获取该层的记忆状态
            layer_memory = None
            if memory_states is not None and i < len(memory_states):
                layer_memory = memory_states[i]
            
            # 直接前向传播（梯度检查点与 Fielix 组件不兼容）
            x, new_layer_memory, aux_loss = layer(
                x, 
                mask=mask,
                memory_state=layer_memory,
                causal=causal
            )
            
            new_memory_states.append(new_layer_memory)
            
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
        
        # 最终归一化
        x = self.final_norm(x)
        
        return x, new_memory_states, total_aux_loss


class FielixLMHead(nn.Module):
    """
    语言模型头：将隐藏状态映射到词汇表
    """
    
    def __init__(self, config: FielixConfig, shared_embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.config = config
        
        if config.tie_word_embeddings and shared_embedding is not None:
            self.weight = shared_embedding.weight
        else:
            self.weight = nn.Parameter(torch.empty(config.vocab_size, config.dim))
            nn.init.normal_(self.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return F.linear(x, self.weight)


class FielixForCausalLM(nn.Module):
    """
    Fielix 因果语言模型
    
    完整的语言模型实现，支持：
    - 自回归生成
    - 流式推理
    - KV 缓存（通过记忆系统实现）
    """
    
    def __init__(self, config: FielixConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embedding = FielixEmbedding(config)
        
        # 解码器
        self.decoder = FielixDecoder(config)
        
        # 语言模型头
        self.lm_head = FielixLMHead(
            config, 
            shared_embedding=self.embedding.token_embedding if config.tie_word_embeddings else None
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        memory_states: Optional[List[List[torch.Tensor]]] = None,
        return_memory: bool = False
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: (batch, seq_len) 输入 token ID
            attention_mask: (batch, seq_len) 注意力掩码
            labels: (batch, seq_len) 标签（用于计算损失）
            memory_states: 记忆状态
            return_memory: 是否返回记忆状态
        
        Returns:
            dict: 包含 logits, loss, memory_states 等
        """
        # 嵌入
        hidden_states = self.embedding(input_ids)
        
        # 解码器
        hidden_states, new_memory_states, aux_loss = self.decoder(
            hidden_states,
            mask=attention_mask,
            memory_states=memory_states,
            causal=True
        )
        
        # 语言模型头
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 移位以计算下一个 token 预测损失
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            
            # 添加辅助损失
            if isinstance(aux_loss, torch.Tensor):
                loss = loss + aux_loss
        
        output = {
            'logits': logits,
            'loss': loss,
            'aux_loss': aux_loss
        }
        
        if return_memory:
            output['memory_states'] = new_memory_states
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        自回归生成（简化版，更稳定）
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        device = input_ids.device
        max_len = self.config.max_seq_len
        
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx = input_ids[:, -max_len:] if input_ids.size(1) > max_len else input_ids
            
            # 前向传播（不使用记忆状态，简化生成）
            outputs = self.forward(idx)
            logits = outputs['logits'][:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-K 采样
            if do_sample and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # 采样或贪婪
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # 检查 EOS
            if next_token.item() == eos_token_id:
                break
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        计算参数数量
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embedding.token_embedding.weight.numel()
        return n_params


class FielixForSequenceClassification(nn.Module):
    """
    Fielix 序列分类模型
    """
    
    def __init__(self, config: FielixConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # 嵌入层
        self.embedding = FielixEmbedding(config)
        
        # 解码器
        self.decoder = FielixDecoder(config)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, num_labels)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch,) 分类标签
        
        Returns:
            dict: 包含 logits, loss
        """
        # 嵌入
        hidden_states = self.embedding(input_ids)
        
        # 解码器（非因果）
        hidden_states, _, aux_loss = self.decoder(
            hidden_states,
            mask=attention_mask,
            causal=False
        )
        
        # 池化（使用第一个 token）
        pooled = hidden_states[:, 0, :]
        
        # 分类
        logits = self.classifier(pooled)
        
        # 损失
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if isinstance(aux_loss, torch.Tensor):
                loss = loss + aux_loss
        
        return {
            'logits': logits,
            'loss': loss
        }


class FielixForTokenClassification(nn.Module):
    """
    Fielix Token 分类模型（NER、POS 等）
    """
    
    def __init__(self, config: FielixConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # 嵌入层
        self.embedding = FielixEmbedding(config)
        
        # 解码器
        self.decoder = FielixDecoder(config)
        
        # 分类头
        self.classifier = nn.Linear(config.dim, num_labels)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) token 级别标签
        
        Returns:
            dict: 包含 logits, loss
        """
        # 嵌入
        hidden_states = self.embedding(input_ids)
        
        # 解码器（非因果）
        hidden_states, _, aux_loss = self.decoder(
            hidden_states,
            mask=attention_mask,
            causal=False
        )
        
        # 分类
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        # 损失
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100
            )
            if isinstance(aux_loss, torch.Tensor):
                loss = loss + aux_loss
        
        return {
            'logits': logits,
            'loss': loss
        }
