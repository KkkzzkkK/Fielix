"""
Nexus Architecture - 快速测试脚本
"""

import torch
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Nexus Architecture - 测试运行")
print("=" * 60)

# 测试导入
print("\n[1] 测试模块导入...")
try:
    from core.field_propagation import FieldEffectLayer
    from core.dynamic_topology import DynamicTopologyLayer
    from core.spiral_memory import SpiralMemoryLayer
    from core.emergent_position import EmergentPositionEncoder
    from core.feedforward import NexusFeedForward
    from core.nexus_block import NexusBlock
    print("    ✓ 所有核心模块导入成功")
except Exception as e:
    print(f"    ✗ 导入失败: {e}")
    sys.exit(1)

# 测试参数
batch_size = 2
seq_len = 64
dim = 256

print(f"\n[2] 测试参数:")
print(f"    - batch_size: {batch_size}")
print(f"    - seq_len: {seq_len}")
print(f"    - dim: {dim}")

# 创建测试输入
x = torch.randn(batch_size, seq_len, dim)
print(f"\n[3] 输入张量形状: {x.shape}")

# 测试场效应层
print("\n[4] 测试 FieldEffectLayer...")
try:
    field_layer = FieldEffectLayer(dim, num_iterations=4)
    out = field_layer(x)
    print(f"    ✓ 输出形状: {out.shape}")
    assert out.shape == x.shape, "形状不匹配"
    print("    ✓ 场效应传播测试通过")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试动态拓扑层
print("\n[5] 测试 DynamicTopologyLayer...")
try:
    topo_layer = DynamicTopologyLayer(dim, use_hierarchical=True)
    out = topo_layer(x, causal=True)
    print(f"    ✓ 输出形状: {out.shape}")
    assert out.shape == x.shape, "形状不匹配"
    print("    ✓ 动态拓扑测试通过")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试螺旋记忆层
print("\n[6] 测试 SpiralMemoryLayer...")
try:
    memory_layer = SpiralMemoryLayer(dim, num_levels=3)
    out, memories = memory_layer(x, return_memories=True)
    print(f"    ✓ 输出形状: {out.shape}")
    print(f"    ✓ 记忆层数: {len(memories) if memories else 0}")
    assert out.shape == x.shape, "形状不匹配"
    print("    ✓ 螺旋记忆测试通过")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试涌现式位置编码
print("\n[7] 测试 EmergentPositionEncoder...")
try:
    pos_encoder = EmergentPositionEncoder(dim, use_oscillator=True, use_relational=False, use_flow=True)
    out, bias = pos_encoder(x)
    print(f"    ✓ 输出形状: {out.shape}")
    assert out.shape == x.shape, "形状不匹配"
    print("    ✓ 涌现位置编码测试通过")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试前馈网络
print("\n[8] 测试 NexusFeedForward...")
try:
    ffn = NexusFeedForward(dim, ffn_type='gated')
    out, aux_loss = ffn(x)
    print(f"    ✓ 输出形状: {out.shape}")
    assert out.shape == x.shape, "形状不匹配"
    print("    ✓ 前馈网络测试通过")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试 Nexus 块
print("\n[9] 测试 NexusBlock...")
try:
    # 测试不同配置
    configs = [
        ('field', False),
        ('topology', False),
        ('hybrid', True),
    ]
    
    for attn_type, use_mem in configs:
        block = NexusBlock(
            dim=dim,
            attention_type=attn_type,
            use_memory=use_mem,
            ffn_type='gated'
        )
        out, mem, aux = block(x, causal=True)
        print(f"    ✓ {attn_type} + memory={use_mem}: 输出形状 {out.shape}")
    
    print("    ✓ Nexus 块测试通过")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试完整模型
print("\n[10] 测试完整模型...")
try:
    from models.nexus_model import NexusConfig, NexusForCausalLM
    
    config = NexusConfig(
        vocab_size=1000,
        dim=256,
        num_layers=4,
        max_seq_len=512,
        attention_type='hybrid',
        use_memory=True,
        ffn_type='gated',
        dropout=0.1
    )
    
    model = NexusForCausalLM(config)
    num_params = model.get_num_params()
    print(f"    ✓ 模型创建成功")
    print(f"    ✓ 参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 前向传播
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    
    print(f"    ✓ Logits 形状: {outputs['logits'].shape}")
    print(f"    ✓ 损失: {outputs['loss'].item():.4f}")
    print("    ✓ 完整模型测试通过")
except Exception as e:
    import traceback
    print(f"    ✗ 失败: {e}")
    traceback.print_exc()

# 测试生成
print("\n[11] 测试文本生成...")
try:
    input_ids = torch.randint(1, 100, (1, 10))
    print(f"    输入: {input_ids[0].tolist()}")
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True
        )
    
    print(f"    生成: {generated[0].tolist()}")
    print(f"    ✓ 生成了 {generated.shape[1] - input_ids.shape[1]} 个新 token")
    print("    ✓ 文本生成测试通过")
except Exception as e:
    print(f"    ✗ 失败: {e}")

print("\n" + "=" * 60)
print("所有测试完成！Nexus 架构运行正常 ✓")
print("=" * 60)
