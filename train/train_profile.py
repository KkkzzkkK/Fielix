"""
原版 Nexus 性能剖析 - 使用 torch.profiler 找出瓶颈
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.profiler
import os
import sys
import json
from pathlib import Path
from datasets import load_dataset

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

sys.path.append(str(Path(__file__).parent.parent))
from models.nexus_model import NexusConfig, NexusForCausalLM

# ============================================================================
# 分词器和数据集
# ============================================================================

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "<USER>": 4, "<BOT>": 5}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = 6
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()+-*/=，。！？、：；""''（）【】《》":
            if c not in self.char_to_id:
                self.char_to_id[c] = self.vocab_size
                self.id_to_char[self.vocab_size] = c
                self.vocab_size += 1
    
    def add_text(self, text):
        for c in text:
            if c not in self.char_to_id:
                self.char_to_id[c] = self.vocab_size
                self.id_to_char[self.vocab_size] = c
                self.vocab_size += 1
    
    def encode(self, text):
        return [1] + [self.char_to_id.get(c, 3) for c in text] + [2]

class ChatDataset(Dataset):
    def __init__(self, data, max_len=128):
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx]
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids[:self.max_len], dtype=torch.long)

# ============================================================================
# 性能剖析
# ============================================================================

def profile_nexus():
    print("=" * 60)
    print("原版 Nexus 性能剖析")
    print("=" * 60)
    
    device = "cuda"
    
    # 加载数据（少量用于剖析）
    print("\n加载 BELLE 数据集...")
    ds = load_dataset("BelleGroup/train_0.5M_CN", split="train[:5000]")
    print(f"数据集大小: {len(ds)}")
    
    tokenizer = CharTokenizer()
    data = []
    
    for i, item in enumerate(ds):
        if 'instruction' in item:
            user = (item['instruction'] + item.get('input', ''))[:200]
            bot = item['output'][:200]
            if user and bot:
                text = f"<USER>{user}<BOT>{bot}"
                tokenizer.add_text(text)
                data.append(tokenizer.encode(text))
    
    print(f"训练样本: {len(data)}")
    print(f"词汇量: {tokenizer.vocab_size}")
    
    max_len = 128
    dataset = ChatDataset(data, max_len)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, 
                           num_workers=4, pin_memory=True, drop_last=True)
    
    # 创建原版 Nexus 模型
    print("\n创建原版 Nexus 模型...")
    config = NexusConfig(
        vocab_size=tokenizer.vocab_size + 100,
        dim=512, num_layers=6, max_seq_len=max_len,
        attention_type='field', use_memory=False, ffn_type='gated',
        dropout=0.1, pad_token_id=0,
    )
    model = NexusForCausalLM(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,} ({params/1e6:.2f}M)")
    
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler("cuda")
    model.train()
    
    # 创建 profiler 输出目录
    os.makedirs("./log/nexus_profile", exist_ok=True)
    
    print("\n开始性能剖析...")
    print("剖析配置: wait=1, warmup=1, active=3, repeat=2")
    
    # 使用 torch.profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/nexus_profile'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        
        for i, batch in enumerate(dataloader):
            if i > 10:  # 只剖析前几个 batch
                break
            
            batch = batch.to(device, non_blocking=True)
            
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(batch, labels=batch)
                loss = outputs['loss']
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 关键：每步调用 profiler.step()
            prof.step()
            
            print(f"  batch {i}: loss = {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("剖析完成！")
    print("=" * 60)
    
    # 打印关键统计
    print("\n=== CPU 时间 Top 10 操作 ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    print("\n=== CUDA 时间 Top 10 操作 ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\n=== Self CUDA 时间 Top 10 操作 ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    
    print("\n查看详细结果:")
    print("  tensorboard --logdir=./log/nexus_profile")

if __name__ == "__main__":
    profile_nexus()
