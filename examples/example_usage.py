"""
Nexus Architecture - 使用示例
演示如何使用 Nexus 架构

本文件展示：
1. 创建模型
2. 前向传播
3. 文本生成
4. 自定义配置
"""

import torch
import sys
sys.path.append('..')

from nexus import (
    NexusConfig,
    NexusForCausalLM,
    NexusForSequenceClassification,
    create_nexus_tiny,
    create_nexus_small,
    create_nexus_base
)


def demo_basic_usage():
    """基本用法演示"""
    print("=" * 60)
    print("Nexus Architecture - 基本用法演示")
    print("=" * 60)
    
    # 创建配置
    config = NexusConfig(
        vocab_size=32000,
        dim=256,
        num_layers=4,
        max_seq_len=512,
        attention_type='hybrid',
        use_memory=True,
        ffn_type='gated',
        dropout=0.1
    )
    
    print(f"\n模型配置:")
    print(f"  - 词汇表大小: {config.vocab_size}")
    print(f"  - 隐藏维度: {config.dim}")
    print(f"  - 层数: {config.num_layers}")
    print(f"  - 注意力类型: {config.attention_type}")
    print(f"  - 使用记忆: {config.use_memory}")
    
    # 创建模型
    model = NexusForCausalLM(config)
    
    # 计算参数量
    num_params = model.get_num_params()
    print(f"\n模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 创建输入
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\n输入形状: {input_ids.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    print(f"输出 logits 形状: {outputs['logits'].shape}")
    print(f"辅助损失: {outputs['aux_loss']}")
    
    return model, config


def demo_text_generation():
    """文本生成演示"""
    print("\n" + "=" * 60)
    print("Nexus Architecture - 文本生成演示")
    print("=" * 60)
    
    # 创建小模型
    model = create_nexus_tiny()
    model.eval()
    
    # 模拟输入（实际应用中需要 tokenizer）
    input_ids = torch.randint(1, 100, (1, 10))
    
    print(f"\n输入序列长度: {input_ids.shape[1]}")
    print(f"输入 token IDs: {input_ids[0].tolist()}")
    
    # 生成
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    
    print(f"\n生成序列长度: {generated.shape[1]}")
    print(f"生成 token IDs: {generated[0].tolist()}")


def demo_sequence_classification():
    """序列分类演示"""
    print("\n" + "=" * 60)
    print("Nexus Architecture - 序列分类演示")
    print("=" * 60)
    
    config = NexusConfig(
        vocab_size=32000,
        dim=256,
        num_layers=4,
        attention_type='topology',
        use_memory=False
    )
    
    num_labels = 5
    model = NexusForSequenceClassification(config, num_labels)
    model.eval()
    
    # 创建输入
    input_ids = torch.randint(0, config.vocab_size, (4, 32))
    labels = torch.randint(0, num_labels, (4,))
    
    print(f"\n输入形状: {input_ids.shape}")
    print(f"类别数: {num_labels}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    
    print(f"分类 logits 形状: {outputs['logits'].shape}")
    print(f"损失: {outputs['loss'].item():.4f}")
    
    # 预测
    predictions = outputs['logits'].argmax(dim=-1)
    print(f"预测结果: {predictions.tolist()}")
    print(f"真实标签: {labels.tolist()}")


def demo_model_variants():
    """模型变体演示"""
    print("\n" + "=" * 60)
    print("Nexus Architecture - 模型变体演示")
    print("=" * 60)
    
    variants = [
        ("Nexus-Tiny", create_nexus_tiny),
        ("Nexus-Small", create_nexus_small),
        ("Nexus-Base", create_nexus_base),
    ]
    
    for name, factory in variants:
        model = factory()
        num_params = model.get_num_params()
        print(f"\n{name}:")
        print(f"  - 参数量: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  - 维度: {model.config.dim}")
        print(f"  - 层数: {model.config.num_layers}")
        print(f"  - 注意力类型: {model.config.attention_type}")


def demo_custom_architecture():
    """自定义架构演示"""
    print("\n" + "=" * 60)
    print("Nexus Architecture - 自定义架构演示")
    print("=" * 60)
    
    # 创建自定义配置
    configs = [
        {
            "name": "Field-Only",
            "config": NexusConfig(
                dim=256, num_layers=4,
                attention_type='field',
                use_memory=False,
                ffn_type='gated'
            )
        },
        {
            "name": "Topology-Only",
            "config": NexusConfig(
                dim=256, num_layers=4,
                attention_type='topology',
                use_memory=False,
                ffn_type='gated'
            )
        },
        {
            "name": "Hybrid + Memory",
            "config": NexusConfig(
                dim=256, num_layers=4,
                attention_type='hybrid',
                use_memory=True,
                ffn_type='gated'
            )
        },
        {
            "name": "MoE Variant",
            "config": NexusConfig(
                dim=256, num_layers=4,
                attention_type='hybrid',
                use_memory=False,
                ffn_type='moe',
                num_experts=4
            )
        },
    ]
    
    for item in configs:
        model = NexusForCausalLM(item["config"])
        num_params = model.get_num_params()
        print(f"\n{item['name']}:")
        print(f"  - 参数量: {num_params:,}")
    
    print("\n可以通过组合不同的注意力类型、记忆配置和前馈网络来创建各种变体。")


def demo_memory_system():
    """记忆系统演示"""
    print("\n" + "=" * 60)
    print("Nexus Architecture - 螺旋记忆系统演示")
    print("=" * 60)
    
    config = NexusConfig(
        dim=256,
        num_layers=4,
        attention_type='field',
        use_memory=True,
        memory_levels=3
    )
    
    model = NexusForCausalLM(config)
    model.eval()
    
    # 模拟流式处理
    print("\n流式处理演示（分块处理长序列）:")
    
    chunk_size = 32
    num_chunks = 4
    memory_states = None
    
    for i in range(num_chunks):
        chunk = torch.randint(0, config.vocab_size, (1, chunk_size))
        
        with torch.no_grad():
            outputs = model(
                chunk,
                memory_states=memory_states,
                return_memory=True
            )
        
        memory_states = outputs['memory_states']
        
        print(f"  块 {i+1}/{num_chunks}: 处理 {chunk_size} tokens, "
              f"记忆状态已更新")
    
    print("\n记忆系统允许模型在处理长序列时保持上下文信息。")


def demo_inference_speed():
    """推理速度演示"""
    print("\n" + "=" * 60)
    print("Nexus Architecture - 推理速度演示")
    print("=" * 60)
    
    import time
    
    model = create_nexus_tiny()
    model.eval()
    
    # 预热
    with torch.no_grad():
        _ = model(torch.randint(0, 100, (1, 32)))
    
    # 测试不同序列长度
    seq_lengths = [64, 128, 256, 512]
    
    print("\n不同序列长度的前向传播时间:")
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 100, (1, seq_len))
        
        # 计时
        start = time.time()
        num_runs = 10
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_ids)
        
        elapsed = (time.time() - start) / num_runs * 1000
        print(f"  序列长度 {seq_len:4d}: {elapsed:.2f} ms/forward")


if __name__ == "__main__":
    # 运行所有演示
    demo_basic_usage()
    demo_text_generation()
    demo_sequence_classification()
    demo_model_variants()
    demo_custom_architecture()
    demo_memory_system()
    demo_inference_speed()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
