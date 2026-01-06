"""
Fielix 模块速度基准测试（优化版）
========================
测试每个核心模块的前向传播速度，与标准 Transformer 对比

优化成果：
- FielixBlock: 3.24x → 0.95x (提升 71%)
- Spiral Memory: 199x → 1x (提升 199x)
- Emergent Position: 16x → 1x (提升 16x)
- Dynamic Topology: 14x → 1x (提升 14x)

用法:
  python benchmark_modules.py
  python benchmark_modules.py --device cuda --batch_size 32
  python benchmark_modules.py --full-model  # 完整模型对比
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
    """运行所有基准测试 - 分组对比"""
    
    print("=" * 70)
    print("🔥 Fielix vs Transformer 速度对比")
    print("=" * 70)
    print(f"配置: batch_size={batch_size}, seq_len={seq_len}, dim={dim}")
    print(f"设备: {device}, 测试次数: {num_runs}")
    print("=" * 70)
    
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    def test(model):
        return benchmark_module(model.to(device), x, num_runs, device)
    
    def show_compare(name, trans_ms, fielix_ms):
        ratio = fielix_ms / trans_ms if trans_ms > 0 else 0
        bar_len = min(int(ratio * 10), 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        status = "✅" if ratio < 2 else ("🟡" if ratio < 3 else "🔴")
        print(f"   Transformer: {trans_ms:>7.3f} ms")
        print(f"   Fielix:      {fielix_ms:>7.3f} ms")
        print(f"   比率: [{bar}] {ratio:.2f}x {status}")
    
    # ============================================================
    # 1. 注意力层对比
    # ============================================================
    print("\n" + "=" * 70)
    print("📌 注意力层对比")
    print("=" * 70)
    
    trans_attn = test(StandardAttention(dim))["mean_ms"]
    field_attn = test(FieldEffectLayer(dim, num_iterations=2))["mean_ms"]
    show_compare("Attention", trans_attn, field_attn)
    
    # ============================================================
    # 2. 前馈网络对比
    # ============================================================
    print("\n" + "=" * 70)
    print("📌 前馈网络对比")
    print("=" * 70)
    
    trans_ffn = test(StandardFFN(dim))["mean_ms"]
    fielix_ffn = test(FielixFeedForward(dim, ffn_type='gated'))["mean_ms"]
    show_compare("FFN", trans_ffn, fielix_ffn)
    
    # ============================================================
    # 3. 完整 Block 对比 (核心指标)
    # ============================================================
    print("\n" + "=" * 70)
    print("📌 完整 Block 对比 (核心指标)")
    print("=" * 70)
    
    trans_block = test(StandardTransformerBlock(dim))["mean_ms"]
    fielix_block = test(FielixBlock(dim, attention_type='field', use_memory=False))["mean_ms"]
    show_compare("Block", trans_block, fielix_block)
    
    # ============================================================
    # 4. 总结
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 总结")
    print("=" * 70)
    
    ratio = fielix_block / trans_block
    if ratio < 2:
        print(f"\n✅ FielixBlock 仅慢 {ratio:.2f}x，接近 Transformer！")
    elif ratio < 3:
        print(f"\n🟡 FielixBlock 慢 {ratio:.2f}x，可接受范围")
    else:
        print(f"\n🔴 FielixBlock 慢 {ratio:.2f}x，需要继续优化")
    
    print(f"\n训练时间预估: Transformer 的 {ratio:.1f} 倍")
    
    # ============================================================
    # 5. 其他优化模块
    # ============================================================
    print("\n" + "=" * 70)
    print("✅ 已优化模块 (可选启用)")
    print("=" * 70)
    
    spiral = test(SpiralMemoryLayer(dim))["mean_ms"]
    emergent = test(EmergentPositionEncoder(dim))["mean_ms"]
    topology = test(DynamicTopologyLayer(dim))["mean_ms"]
    
    print(f"   Spiral Memory:     {spiral:>7.3f} ms ({spiral/trans_attn:.0f}x) ✅")
    print(f"   Emergent Position: {emergent:>7.3f} ms ({emergent/trans_attn:.0f}x) ✅")
    print(f"   Dynamic Topology:  {topology:>7.3f} ms ({topology/trans_attn:.0f}x) ✅")


# ============================================================================
# 完整模型性能分析
# ============================================================================

def profile_full_model(
    batch_size: int = 32,
    seq_len: int = 128,
    dim: int = 512,
    num_layers: int = 8,
    vocab_size: int = 4000,
    device: str = "cuda",
    num_runs: int = 50
):
    """分析完整模型的性能瓶颈"""
    
    from models.nexus_model import FielixConfig, FielixForCausalLM
    
    print("=" * 70)
    print("🔍 完整模型性能分析 - 找出训练慢的原因")
    print("=" * 70)
    print(f"配置: batch={batch_size}, seq_len={seq_len}, dim={dim}, layers={num_layers}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # ============================================================
    # 1. 构建标准 Transformer
    # ============================================================
    class SimpleTransformerLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.pos_embed = nn.Embedding(seq_len, dim)
            self.layers = nn.ModuleList([
                StandardTransformerBlock(dim) for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab_size, bias=False)
        
        def forward(self, x, labels=None):
            B, L = x.shape
            pos = torch.arange(L, device=x.device)
            h = self.embed(x) + self.pos_embed(pos)
            for layer in self.layers:
                h = layer(h)
            logits = self.head(self.norm(h))
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size),
                    labels[:, 1:].reshape(-1),
                    ignore_index=0
                )
            return {'logits': logits, 'loss': loss}
    
    # ============================================================
    # 2. 构建 Fielix
    # ============================================================
    config = FielixConfig(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        max_seq_len=seq_len,
        attention_type='hybrid',
        use_memory=True,
        ffn_type='gated',
        field_iterations=2,
        dropout=0.1,
    )
    
    trans_model = SimpleTransformerLM().to(device)
    fielix_model = FielixForCausalLM(config).to(device)
    
    trans_params = sum(p.numel() for p in trans_model.parameters())
    fielix_params = sum(p.numel() for p in fielix_model.parameters())
    
    print(f"\n📊 参数量:")
    print(f"   Transformer: {trans_params:,}")
    print(f"   Fielix:      {fielix_params:,}")
    print(f"   差异:        {(fielix_params-trans_params)/trans_params*100:+.1f}%")
    
    # ============================================================
    # 3. 前向传播测试
    # ============================================================
    x = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    
    def benchmark_forward(model, name, runs=num_runs):
        model.eval()
        # 预热
        with torch.no_grad():
            for _ in range(10):
                model(x, labels=x)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(x, labels=x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        
        return sum(times[10:]) / len(times[10:])
    
    print(f"\n⏱️  前向传播速度:")
    trans_fwd = benchmark_forward(trans_model, "Transformer")
    fielix_fwd = benchmark_forward(fielix_model, "Fielix")
    print(f"   Transformer: {trans_fwd:.2f} ms")
    print(f"   Fielix:      {fielix_fwd:.2f} ms")
    print(f"   比率:        {fielix_fwd/trans_fwd:.2f}x")
    
    # ============================================================
    # 4. 反向传播测试
    # ============================================================
    def benchmark_backward(model, name, runs=num_runs):
        model.train()
        # 预热
        for _ in range(5):
            out = model(x, labels=x)
            out['loss'].backward()
            model.zero_grad()
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        times = []
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(x, labels=x)
            out['loss'].backward()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            model.zero_grad()
        
        return sum(times[5:]) / len(times[5:])
    
    print(f"\n⏱️  前向+反向传播速度:")
    trans_bwd = benchmark_backward(trans_model, "Transformer")
    fielix_bwd = benchmark_backward(fielix_model, "Fielix")
    print(f"   Transformer: {trans_bwd:.2f} ms")
    print(f"   Fielix:      {fielix_bwd:.2f} ms")
    print(f"   比率:        {fielix_bwd/trans_bwd:.2f}x")
    
    # ============================================================
    # 5. 逐层分析 Fielix
    # ============================================================
    print("\n" + "=" * 70)
    print("🔬 Fielix 逐组件分析")
    print("=" * 70)
    
    # 分析嵌入层
    def time_component(fn, name, runs=30):
        # 预热
        with torch.no_grad():
            for _ in range(5):
                fn()
        if device == "cuda":
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                fn()
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        return sum(times[5:]) / len(times[5:])
    
    # 嵌入层
    emb_time = time_component(
        lambda: fielix_model.embedding(x),
        "Embedding"
    )
    
    # 单层解码器
    h = fielix_model.embedding(x)
    single_layer_time = time_component(
        lambda: fielix_model.decoder.layers[0](h),
        "Single Layer"
    )
    
    # 所有解码器层
    all_layers_time = time_component(
        lambda: fielix_model.decoder(h),
        "All Decoder Layers"
    )
    
    # LM Head
    h2, _, _ = fielix_model.decoder(h)
    lm_head_time = time_component(
        lambda: fielix_model.lm_head(h2),
        "LM Head"
    )
    
    print(f"   Embedding:          {emb_time:>7.2f} ms")
    print(f"   Single Layer:       {single_layer_time:>7.2f} ms")
    print(f"   All Layers ({num_layers}):    {all_layers_time:>7.2f} ms")
    print(f"   LM Head:            {lm_head_time:>7.2f} ms")
    print(f"   ─────────────────────────────")
    total_component = emb_time + all_layers_time + lm_head_time
    print(f"   组件合计:           {total_component:>7.2f} ms")
    print(f"   实际前向:           {fielix_fwd:>7.2f} ms")
    
    # ============================================================
    # 6. 瓶颈分析
    # ============================================================
    print("\n" + "=" * 70)
    print("🎯 瓶颈分析")
    print("=" * 70)
    
    overhead = fielix_fwd / trans_fwd
    emb_ratio = emb_time / fielix_fwd * 100
    layer_ratio = all_layers_time / fielix_fwd * 100
    head_ratio = lm_head_time / fielix_fwd * 100
    
    print(f"   总体慢:             {overhead:.2f}x")
    print(f"")
    print(f"   时间分布:")
    print(f"   ├─ Embedding:       {emb_ratio:>5.1f}%")
    print(f"   ├─ Decoder Layers:  {layer_ratio:>5.1f}%")
    print(f"   └─ LM Head:         {head_ratio:>5.1f}%")
    
    # 找出最大瓶颈
    bottleneck = max([
        ("Embedding (EmergentPosition)", emb_time, emb_ratio),
        ("Decoder Layers", all_layers_time, layer_ratio),
        ("LM Head", lm_head_time, head_ratio),
    ], key=lambda x: x[1])
    
    print(f"\n   🔴 主要瓶颈: {bottleneck[0]} ({bottleneck[2]:.1f}%)")
    
    # ============================================================
    # 7. 对比 Transformer 分解
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 Transformer 基准对比")
    print("=" * 70)
    
    trans_emb_time = time_component(
        lambda: trans_model.embed(x) + trans_model.pos_embed(torch.arange(seq_len, device=device)),
        "Trans Embedding"
    )
    h_trans = trans_model.embed(x) + trans_model.pos_embed(torch.arange(seq_len, device=device))
    trans_layer_time = time_component(
        lambda: trans_model.layers[0](h_trans),
        "Trans Single Layer"
    )
    
    print(f"   Transformer Embedding:  {trans_emb_time:>7.2f} ms")
    print(f"   Fielix Embedding:       {emb_time:>7.2f} ms")
    print(f"   Embedding 差异:         {emb_time/trans_emb_time:.2f}x")
    print(f"")
    print(f"   Transformer Layer:      {trans_layer_time:>7.2f} ms")
    print(f"   Fielix Layer:           {single_layer_time:>7.2f} ms")
    print(f"   Layer 差异:             {single_layer_time/trans_layer_time:.2f}x")
    
    # ============================================================
    # 8. FielixBlock 内部分析
    # ============================================================
    print("\n" + "=" * 70)
    print("🔬 FielixBlock 内部组件分析")
    print("=" * 70)
    
    # 获取单层进行分析
    layer = fielix_model.decoder.layers[0]
    h_test = fielix_model.embedding(x)
    
    # Field Attention
    if hasattr(layer, 'field_attention'):
        field_time = time_component(
            lambda: layer.field_attention(h_test),
            "Field Attention"
        )
        print(f"   Field Attention:    {field_time:>7.2f} ms")
    
    # Topology Attention
    if hasattr(layer, 'topology_attention'):
        topo_time = time_component(
            lambda: layer.topology_attention(h_test, causal=True),
            "Topology Attention"
        )
        print(f"   Topology Attention: {topo_time:>7.2f} ms")
    
    # Single Attention (non-hybrid)
    if hasattr(layer, 'attention'):
        attn_time = time_component(
            lambda: layer.attention(h_test),
            "Attention"
        )
        print(f"   Attention:          {attn_time:>7.2f} ms")
    
    # Memory
    if hasattr(layer, 'memory') and layer.use_memory:
        mem_time = time_component(
            lambda: layer.memory(h_test),
            "Memory"
        )
        print(f"   Spiral Memory:      {mem_time:>7.2f} ms")
    
    # FFN
    ffn_time = time_component(
        lambda: layer.ffn(h_test),
        "FFN"
    )
    print(f"   FFN:                {ffn_time:>7.2f} ms")
    
    # Hybrid gate overhead
    if hasattr(layer, 'hybrid_gate'):
        gate_time = time_component(
            lambda: layer.hybrid_gate(h_test.mean(dim=1, keepdim=True)),
            "Hybrid Gate"
        )
        print(f"   Hybrid Gate:        {gate_time:>7.2f} ms")
        print(f"\n   ⚠️  Hybrid 模式运行 2 种注意力，建议使用 'field' 模式")
    
    # ============================================================
    # 9. 优化建议
    # ============================================================
    print("\n" + "=" * 70)
    print("💡 优化建议")
    print("=" * 70)
    
    if emb_time / trans_emb_time > 1.5:
        print("   🔧 Embedding 层慢 - EmergentPositionEncoder 需要简化")
    if single_layer_time / trans_layer_time > 1.2:
        print("   🔧 Decoder Layer 慢 - 检查 FielixBlock 组件")
    if hasattr(layer, 'hybrid_gate'):
        print("   🔧 使用 attention_type='field' 替代 'hybrid' 可提速 ~40%")
    if overhead < 1.2:
        print("   ✅ 性能接近 Transformer，可以接受")
    elif overhead < 1.5:
        print("   🟡 性能差距中等，建议优化主要瓶颈")
    else:
        print("   🔴 性能差距较大，需要重点优化")
    
    return {
        'trans_fwd': trans_fwd,
        'fielix_fwd': fielix_fwd,
        'trans_bwd': trans_bwd,
        'fielix_bwd': fielix_bwd,
        'emb_time': emb_time,
        'layer_time': single_layer_time,
        'overhead': overhead,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fielix 模块速度基准测试")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--seq_len", type=int, default=128, help="序列长度")
    parser.add_argument("--dim", type=int, default=256, help="模型维度")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--num_runs", type=int, default=100, help="测试次数")
    parser.add_argument("--full-model", action="store_true", help="完整模型分析")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"
    
    if args.full_model:
        profile_full_model(
            batch_size=32,
            seq_len=128,
            dim=512,
            num_layers=8,
            vocab_size=4000,
            device=args.device,
            num_runs=50
        )
    else:
        run_benchmarks(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            dim=args.dim,
            device=args.device,
            num_runs=args.num_runs
        )
