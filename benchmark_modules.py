"""
Fielix 模块速度基准测试
========================
测试每个核心模块的前向传播速度，与标准 Transformer 对比

用法:
  python benchmark_modules.py
  python benchmark_modules.py --device cuda --batch_size 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math
from typing import Dict, List, Tuple

# 导入 Fielix 核心模块
from core.field_propagation import FieldEffectLayer
from core.dynamic_topology import DynamicTopologyLayer
from core.spiral_memory import SpiralMemoryLayer
from core.emergent_position import EmergentPositionEncoder
from core.feedforward import FielixFeedForward
from core.nexus_block import FielixBlock


# ============================================================================
# 标准 Transformer 组件（对照基准）
# ============================================================================

class StandardAttention(nn.Module):
    """标准多头自注意力"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x B x H x L x D
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class StandardFFN(nn.Module):
    """标准前馈网络"""
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class StandardTransformerBlock(nn.Module):
    """标准 Transformer 块"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = StandardAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = StandardFFN(dim, dropout=dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# 基准测试
# ============================================================================

def warmup(model: nn.Module, x: torch.Tensor, num_warmup: int = 5):
    """预热模型"""
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            try:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
            except:
                pass


def benchmark_module(
    model: nn.Module,
    x: torch.Tensor,
    num_runs: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """测试模块速度"""
    model.eval()
    
    # 预热
    warmup(model, x, num_warmup=10)
    
    # 同步 CUDA
    if device == "cuda":
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            try:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
            except Exception as e:
                return {"error": str(e)}
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 毫秒
    
    times = times[10:]  # 去掉前10次
    
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5
    }


def run_benchmarks(
    batch_size: int = 16,
    seq_len: int = 128,
    dim: int = 256,
    device: str = "cuda",
    num_runs: int = 100
):
    """运行所有基准测试"""
    
    print("=" * 70)
    print("Fielix 模块速度基准测试")
    print("=" * 70)
    print(f"配置: batch_size={batch_size}, seq_len={seq_len}, dim={dim}")
    print(f"设备: {device}")
    print(f"测试次数: {num_runs}")
    print("=" * 70)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # 定义要测试的模块
    modules = {}
    
    # 1. 标准组件（基准）
    modules["Transformer Attention"] = StandardAttention(dim).to(device)
    modules["Transformer FFN"] = StandardFFN(dim).to(device)
    modules["Transformer Block"] = StandardTransformerBlock(dim).to(device)
    
    # 2. Fielix 核心模块
    modules["Field Effect (iter=4)"] = FieldEffectLayer(dim, num_iterations=4).to(device)
    modules["Field Effect (iter=2)"] = FieldEffectLayer(dim, num_iterations=2).to(device)
    modules["Dynamic Topology"] = DynamicTopologyLayer(dim).to(device)
    modules["Emergent Position"] = EmergentPositionEncoder(dim).to(device)
    modules["Spiral Memory"] = SpiralMemoryLayer(dim).to(device)
    modules["Fielix FFN (gated)"] = FielixFeedForward(dim, ffn_type='gated').to(device)
    modules["Fielix FFN (moe)"] = FielixFeedForward(dim, ffn_type='moe', num_experts=4).to(device)
    
    # 3. Fielix Block 不同配置
    modules["FielixBlock (field)"] = FielixBlock(dim, attention_type='field', use_memory=False).to(device)
    modules["FielixBlock (field+mem)"] = FielixBlock(dim, attention_type='field', use_memory=True).to(device)
    modules["FielixBlock (topology)"] = FielixBlock(dim, attention_type='topology', use_memory=False).to(device)
    modules["FielixBlock (hybrid)"] = FielixBlock(dim, attention_type='hybrid', use_memory=True).to(device)
    
    # 运行测试
    results = {}
    baseline_time = None
    
    print(f"\n{'模块名称':<30} {'平均(ms)':<12} {'最小(ms)':<12} {'相对速度':<12}")
    print("-" * 70)
    
    for name, module in modules.items():
        result = benchmark_module(module, x, num_runs, device)
        results[name] = result
        
        if "error" in result:
            print(f"{name:<30} ERROR: {result['error']}")
            continue
        
        # 第一个作为基准
        if baseline_time is None:
            baseline_time = result["mean_ms"]
        
        relative = result["mean_ms"] / baseline_time
        speed_indicator = "🟢" if relative < 1.5 else ("🟡" if relative < 3 else "🔴")
        
        print(f"{name:<30} {result['mean_ms']:>8.3f} ms  {result['min_ms']:>8.3f} ms  {relative:>6.2f}x {speed_indicator}")
    
    # 总结
    print("\n" + "=" * 70)
    print("速度分析")
    print("=" * 70)
    
    # 找出最慢的模块
    valid_results = [(n, r) for n, r in results.items() if "error" not in r]
    sorted_results = sorted(valid_results, key=lambda x: x[1]["mean_ms"], reverse=True)
    
    print("\n最慢的 5 个模块：")
    for i, (name, result) in enumerate(sorted_results[:5]):
        relative = result["mean_ms"] / baseline_time
        print(f"  {i+1}. {name}: {result['mean_ms']:.3f} ms ({relative:.2f}x)")
    
    print("\n最快的 5 个模块：")
    for i, (name, result) in enumerate(sorted_results[-5:][::-1]):
        relative = result["mean_ms"] / baseline_time
        print(f"  {i+1}. {name}: {result['mean_ms']:.3f} ms ({relative:.2f}x)")
    
    # 优化建议
    print("\n" + "=" * 70)
    print("优化建议")
    print("=" * 70)
    
    field_iter4 = results.get("Field Effect (iter=4)", {}).get("mean_ms", 0)
    field_iter2 = results.get("Field Effect (iter=2)", {}).get("mean_ms", 0)
    if field_iter4 > 0 and field_iter2 > 0:
        improvement = (field_iter4 - field_iter2) / field_iter4 * 100
        print(f"1. 减少 Field Effect 迭代次数 4→2 可提速 {improvement:.1f}%")
    
    block_field = results.get("FielixBlock (field)", {}).get("mean_ms", 0)
    block_mem = results.get("FielixBlock (field+mem)", {}).get("mean_ms", 0)
    if block_field > 0 and block_mem > 0:
        overhead = (block_mem - block_field) / block_field * 100
        print(f"2. 螺旋记忆开销约 {overhead:.1f}%")
    
    trans_block = results.get("Transformer Block", {}).get("mean_ms", 0)
    fielix_block = results.get("FielixBlock (field)", {}).get("mean_ms", 0)
    if trans_block > 0 and fielix_block > 0:
        slowdown = fielix_block / trans_block
        print(f"3. FielixBlock 相比 Transformer Block 慢 {slowdown:.2f}x")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fielix 模块速度基准测试")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--seq_len", type=int, default=128, help="序列长度")
    parser.add_argument("--dim", type=int, default=256, help="模型维度")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--num_runs", type=int, default=100, help="测试次数")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"
    
    run_benchmarks(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        device=args.device,
        num_runs=args.num_runs
    )
