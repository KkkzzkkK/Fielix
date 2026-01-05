"""
Nexus Architecture - Utilities Module
工具模块

包含：
- 统一的权重初始化策略
- 数值稳定性工具
- 梯度检查点支持
- 内存优化工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, Callable, Any


# ============================================================================
# 统一初始化策略
# ============================================================================

def init_weights(module: nn.Module, std: float = 0.02):
    """
    统一的权重初始化策略
    
    Args:
        module: 要初始化的模块
        std: 标准差
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=std)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def init_nexus_model(model: nn.Module, std: float = 0.02, residual_scale: float = 1.0):
    """
    初始化整个 Nexus 模型
    
    Args:
        model: 模型
        std: 基础标准差
        residual_scale: 残差连接的缩放因子
    """
    num_layers = sum(1 for _ in model.modules() if hasattr(_, 'is_nexus_block'))
    
    for name, module in model.named_modules():
        init_weights(module, std)
        
        # 对残差路径上的输出投影进行特殊初始化
        if 'output_proj' in name or 'out_proj' in name:
            if isinstance(module, nn.Linear):
                # 按层数缩放，确保训练稳定
                scale = residual_scale / math.sqrt(max(1, num_layers))
                nn.init.trunc_normal_(module.weight, std=std * scale)


# ============================================================================
# 数值稳定性工具
# ============================================================================

def stable_softmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    数值稳定的 softmax
    
    Args:
        x: 输入张量
        dim: softmax 的维度
        eps: 防止除零的小常数
    
    Returns:
        softmax 结果
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    return exp_x / (exp_x.sum(dim=dim, keepdim=True) + eps)


def stable_log_softmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    数值稳定的 log_softmax
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    log_sum_exp = torch.log(torch.exp(x_stable).sum(dim=dim, keepdim=True) + eps)
    return x_stable - log_sum_exp


def clamp_gradients(tensor: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
    """
    在前向传播中应用梯度裁剪（通过 hook）
    
    Args:
        tensor: 输入张量
        max_norm: 最大梯度范数
    
    Returns:
        带梯度裁剪 hook 的张量
    """
    if tensor.requires_grad:
        def hook(grad):
            grad_norm = grad.norm()
            if grad_norm > max_norm:
                return grad * (max_norm / grad_norm)
            return grad
        tensor.register_hook(hook)
    return tensor


class GradientClipping(nn.Module):
    """
    梯度裁剪模块 - 可以插入到模型中
    """
    
    def __init__(self, max_norm: float = 1.0):
        super().__init__()
        self.max_norm = max_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.requires_grad:
            return clamp_gradients(x, self.max_norm)
        return x


class NumericallyStableLayerNorm(nn.Module):
    """
    数值稳定的 LayerNorm
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用更稳定的计算方式
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class RMSNorm(nn.Module):
    """
    RMS Normalization - 比 LayerNorm 更快更稳定
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# 梯度检查点支持
# ============================================================================

class CheckpointedModule(nn.Module):
    """
    支持梯度检查点的模块包装器
    
    用法：
        layer = CheckpointedModule(SomeLayer(), use_checkpoint=True)
    """
    
    def __init__(self, module: nn.Module, use_checkpoint: bool = False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint
    
    def forward(self, *args, **kwargs):
        if self.use_checkpoint and self.training:
            # 梯度检查点需要所有输入都是张量
            return checkpoint(self._forward_impl, *args, use_reentrant=False, **kwargs)
        return self.module(*args, **kwargs)
    
    def _forward_impl(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def apply_gradient_checkpointing(model: nn.Module, checkpoint_ratio: float = 0.5):
    """
    对模型应用梯度检查点
    
    Args:
        model: 模型
        checkpoint_ratio: 对多少比例的层应用检查点 (0-1)
    """
    # 找到所有 NexusBlock
    blocks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'NexusBlock':
            blocks.append((name, module))
    
    # 对指定比例的层启用检查点
    num_checkpoint = int(len(blocks) * checkpoint_ratio)
    for i, (name, block) in enumerate(blocks):
        if i < num_checkpoint:
            block.use_gradient_checkpoint = True


# ============================================================================
# 内存优化工具
# ============================================================================

def clear_memory():
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class MemoryEfficientAttention(nn.Module):
    """
    内存高效的注意力实现
    
    使用分块计算避免存储完整的注意力矩阵
    """
    
    def __init__(self, dim: int, num_heads: int = 8, chunk_size: int = 256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 尝试使用 Flash Attention (如果可用)
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal
            )
        else:
            # 回退到分块注意力
            out = self._chunked_attention(q, k, v, causal)
        
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.out_proj(out)
    
    def _chunked_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        causal: bool
    ) -> torch.Tensor:
        """分块计算注意力，减少内存使用"""
        batch, heads, seq_len, head_dim = q.shape
        
        # 如果序列较短，直接计算
        if seq_len <= self.chunk_size:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(causal_mask, float('-inf'))
            attn = stable_softmax(scores, dim=-1)
            return torch.matmul(attn, v)
        
        # 分块计算
        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            chunk_end = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:chunk_end, :]
            
            # 计算与所有 k 的注意力
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            if causal:
                # 只注意当前位置之前的
                chunk_len = chunk_end - i
                causal_mask = torch.ones(chunk_len, seq_len, device=q.device, dtype=torch.bool)
                for j in range(chunk_len):
                    causal_mask[j, i+j+1:] = True
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn = stable_softmax(scores, dim=-1)
            chunk_out = torch.matmul(attn, v)
            outputs.append(chunk_out)
        
        return torch.cat(outputs, dim=2)


# ============================================================================
# 训练工具
# ============================================================================

class EMA:
    """
    指数移动平均
    
    用于模型参数的 EMA 更新，提高训练稳定性
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新 EMA 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用 EMA 参数（用于评估）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class WarmupCosineScheduler:
    """
    带预热的余弦退火学习率调度器
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr * (base_lr / self.base_lrs[0])
    
    def _compute_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # 线性预热
            return self.base_lrs[0] * self.current_step / self.warmup_steps
        else:
            # 余弦退火
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))
