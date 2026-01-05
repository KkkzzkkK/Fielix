"""
Nexus vs Transformer 对比 - 使用专业数据集 (BELLE/ShareGPT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import os
import sys
import json
import random
import time
from pathlib import Path
from datasets import load_dataset

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

sys.path.append(str(Path(__file__).parent.parent))
from models.nexus_model import NexusConfig, NexusForCausalLM

# ============================================================================
# 标准 Transformer 模型
# ============================================================================

class TransformerConfig:
    def __init__(self, vocab_size=500, dim=256, num_layers=4, num_heads=8, 
                 max_seq_len=128, dropout=0.1, pad_token_id=0):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_token_id = pad_token_id

class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.dim)
        self.drop = nn.Dropout(config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim, nhead=config.num_heads,
            dim_feedforward=config.dim * 4, dropout=config.dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
    
    def forward(self, input_ids, labels=None):
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.drop(x)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.head(self.norm(x))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size),
                                   labels[:, 1:].reshape(-1), ignore_index=self.config.pad_token_id)
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=30, temperature=0.7, top_k=30):
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.config.max_seq_len:]
            logits = self(idx)['logits'][:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == 2:
                break
        return input_ids

# ============================================================================
# 分词器
# ============================================================================

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "<USER>": 4, "<BOT>": 5}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = 6
        # 预先添加常用字符
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
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '') for i in ids if i > 5)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'char_to_id': self.char_to_id}, f, ensure_ascii=False, indent=2)

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
# 训练
# ============================================================================

def train_model(model, dataloader, epochs, device, model_name):
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # 大模型用小学习率
    scaler = GradScaler("cuda")
    model.train()
    history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch in dataloader:
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
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        history.append(avg_loss)
        print(f"  [{model_name}] Epoch {epoch+1:2d}/{epochs}: loss = {avg_loss:.4f}")
    
    return history, time.time() - start_time

def test_model(model, tokenizer, device, model_name):
    model.eval()
    tests = ["你好", "介绍一下自己", "1+1等于多少"]
    print(f"\n[{model_name}] 测试:")
    for q in tests:
        ids = tokenizer.encode(f"<USER>{q}<BOT>")[:-1]
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=30, temperature=0.7)
        resp = tokenizer.decode(out[0].tolist())
        if "<BOT>" in resp:
            resp = resp.split("<BOT>")[-1]
        print(f"    {q} -> {resp[:40]}")

def train():
    print("=" * 60)
    print("Nexus vs Transformer - 专业数据集对比")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 直接使用 BELLE 数据集（更稳定）
    print("\n加载 BELLE 数据集...", flush=True)
    ds = load_dataset("BelleGroup/train_0.5M_CN", split="train[:50000]")
    print(f"数据集大小: {len(ds)}")
    
    # 准备数据
    tokenizer = CharTokenizer()
    data = []
    max_samples = 50000  # 限制样本数
    
    print("处理数据...")
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        
        # 根据数据集格式处理
        if 'conversations' in item:
            # ShareGPT 格式
            convs = item['conversations']
            for j in range(0, len(convs) - 1, 2):
                if j + 1 < len(convs):
                    user = convs[j].get('value', convs[j].get('content', ''))[:200]
                    bot = convs[j+1].get('value', convs[j+1].get('content', ''))[:200]
                    if user and bot:
                        text = f"<USER>{user}<BOT>{bot}"
                        tokenizer.add_text(text)
                        data.append(tokenizer.encode(text))
        elif 'instruction' in item:
            # Alpaca 格式
            user = (item['instruction'] + item.get('input', ''))[:200]
            bot = item['output'][:200]
            if user and bot:
                text = f"<USER>{user}<BOT>{bot}"
                tokenizer.add_text(text)
                data.append(tokenizer.encode(text))
        
        if i % 10000 == 0:
            print(f"  处理 {i}/{max_samples}...")
    
    print(f"训练样本: {len(data)}")
    print(f"词汇量: {tokenizer.vocab_size}")
    
    # 数据集（大模型需要小 batch）
    max_len = 128
    dataset = ChatDataset(data, max_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,  # 减小 batch 避免 OOM
                           num_workers=4, pin_memory=True, drop_last=True)
    
    # ============ 训练配置 ============
    epochs = 15
    vocab_size = tokenizer.vocab_size + 100
    os.makedirs("./checkpoints", exist_ok=True)
    
    # ============ 先测试 Transformer ============
    print("\n创建 Transformer 模型 (500M)...")
    trans_config = TransformerConfig(
        vocab_size=vocab_size,
        dim=1024, num_layers=16, num_heads=16, max_seq_len=max_len,
        dropout=0.1, pad_token_id=0,
    )
    trans_model = TransformerLM(trans_config).to(device)
    trans_params = sum(p.numel() for p in trans_model.parameters())
    print(f"Transformer 参数量: {trans_params:,} ({trans_params/1e6:.2f}M)")
    
    print(f"\n开始训练 Transformer ({epochs} epochs)...")
    trans_history, trans_time = train_model(trans_model, dataloader, epochs, device, "Transformer")
    
    # 测试 Transformer
    test_model(trans_model, tokenizer, device, "Transformer")
    
    # 保存 Transformer
    torch.save({'model': trans_model.state_dict(), 'config': trans_config}, "./checkpoints/transformer_belle.pt")
    tokenizer.save("./checkpoints/belle_tokenizer.json")
    
    # ============ 结果 ============
    print("\n" + "=" * 60)
    print("Transformer 500M 测试结果")
    print("=" * 60)
    print(f"参数量: {trans_params:,} ({trans_params/1e6:.2f}M)")
    print(f"最终 Loss: {trans_history[-1]:.4f}")
    print(f"训练时间: {trans_time:.1f}s")
    print("\n模型已保存！")

if __name__ == "__main__":
    train()
