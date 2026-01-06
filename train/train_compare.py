"""
Fielix vs Transformer 对比训练
==============================
用法:
  python train_compare.py --model fielix
  python train_compare.py --model transformer
  python train_compare.py --model both
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import os
import sys
import json
import time
import math
from pathlib import Path
from datasets import load_dataset

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

sys.path.append(str(Path(__file__).parent.parent))
from models.nexus_model import FielixConfig, FielixForCausalLM

# ============================================================================
# 统一配置
# ============================================================================
CONFIG = {
    'batch_size': 256,
    'max_len': 128,
    'epochs': 15,
    'lr': 3e-4,
    'dim': 512,
    'num_layers': 8,
    'num_heads': 8,
    'dropout': 0.1,
    'max_samples': 50000,
    'num_workers': 4,
}

# ============================================================================
# 标准 Transformer
# ============================================================================
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, max_len, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # 权重绑定
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.drop(x)
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.head(self.norm(x))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=0
            )
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=30, temperature=0.7):
        for _ in range(max_new_tokens):
            logits = self(input_ids[:, -CONFIG['max_len']:])['logits'][:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == 2:  # EOS
                break
        return input_ids


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        B, L, D = x.shape
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        
        # Pre-LN
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# 分词器
# ============================================================================
class CharTokenizer:
    def __init__(self):
        self.char_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = 4
    
    def add_text(self, text):
        for c in text:
            if c not in self.char_to_id:
                self.char_to_id[c] = self.vocab_size
                self.id_to_char[self.vocab_size] = c
                self.vocab_size += 1
    
    def encode(self, text):
        return [1] + [self.char_to_id.get(c, 3) for c in text] + [2]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '') for i in ids if i > 3)


class ChatDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx][:self.max_len]
        ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


# ============================================================================
# 训练
# ============================================================================
def train_model(model, dataloader, epochs, device, name):
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scaler = GradScaler("cuda")
    model.train()
    history = []
    start = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for batch in dataloader:
            batch = batch.to(device)
            with autocast(device_type="cuda", dtype=torch.float16):
                out = model(batch, labels=batch)
                loss = out['loss']
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            count += 1
        
        avg = total_loss / count
        history.append(avg)
        print(f"  [{name}] Epoch {epoch+1:2d}/{epochs}: loss = {avg:.4f}")
    
    return history, time.time() - start


def test_model(model, tokenizer, device, name):
    model.eval()
    tests = ["你好", "1+1等于多少", "介绍一下自己"]
    print(f"\n[{name}] 测试:")
    for q in tests:
        ids = tokenizer.encode(q)[:-1]
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=30)
        resp = tokenizer.decode(out[0].tolist())
        print(f"    {q} -> {resp[:50]}")


def main(model_type):
    print("=" * 60)
    print(f"Fielix vs Transformer 对比 (模式: {model_type})")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"配置: {CONFIG}")
    
    # 加载数据
    print("\n加载 BELLE 数据集...")
    ds = load_dataset("BelleGroup/train_0.5M_CN", split=f"train[:{CONFIG['max_samples']}]")
    print(f"数据集: {len(ds)} 条")
    
    tokenizer = CharTokenizer()
    data = []
    
    for item in ds:
        if 'instruction' in item:
            user = (item['instruction'] + item.get('input', ''))[:150]
            bot = item['output'][:150]
            if user and bot:
                text = f"{user}{bot}"
                tokenizer.add_text(text)
                data.append(tokenizer.encode(text))
    
    print(f"样本数: {len(data)}, 词汇量: {tokenizer.vocab_size}")
    
    dataset = ChatDataset(data, CONFIG['max_len'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                           num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    
    vocab_size = tokenizer.vocab_size + 100
    os.makedirs("./checkpoints", exist_ok=True)
    
    fielix_result = None
    trans_result = None
    
    # Fielix
    if model_type in ['fielix', 'both']:
        print("\n" + "=" * 60)
        print("训练 Fielix")
        print("=" * 60)
        
        config = FielixConfig(
            vocab_size=vocab_size,
            dim=CONFIG['dim'],
            num_layers=CONFIG['num_layers'],
            max_seq_len=CONFIG['max_len'],
            attention_type='field',
            use_memory=False,
            ffn_type='gated',
            field_iterations=2,
            dropout=CONFIG['dropout'],
            pad_token_id=0,
        )
        model = FielixForCausalLM(config).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,} ({params/1e6:.2f}M)")
        
        history, train_time = train_model(model, dataloader, CONFIG['epochs'], device, "Fielix")
        test_model(model, tokenizer, device, "Fielix")
        torch.save({'model': model.state_dict()}, "./checkpoints/fielix.pt")
        
        fielix_result = {'params': params, 'loss': history[-1], 'time': train_time}
        
        del model
        torch.cuda.empty_cache()
    
    # Transformer
    if model_type in ['transformer', 'both']:
        print("\n" + "=" * 60)
        print("训练 Transformer")
        print("=" * 60)
        
        model = TransformerLM(
            vocab_size=vocab_size,
            dim=CONFIG['dim'],
            num_layers=CONFIG['num_layers'],
            num_heads=CONFIG['num_heads'],
            max_len=CONFIG['max_len'],
            dropout=CONFIG['dropout'],
        ).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,} ({params/1e6:.2f}M)")
        
        history, train_time = train_model(model, dataloader, CONFIG['epochs'], device, "Transformer")
        test_model(model, tokenizer, device, "Transformer")
        torch.save({'model': model.state_dict()}, "./checkpoints/transformer.pt")
        
        trans_result = {'params': params, 'loss': history[-1], 'time': train_time}
    
    # 对比
    if model_type == 'both' and fielix_result and trans_result:
        print("\n" + "=" * 60)
        print("对比结果")
        print("=" * 60)
        print(f"{'指标':<15} {'Fielix':<20} {'Transformer':<20}")
        print("-" * 55)
        print(f"{'参数量':<15} {fielix_result['params']:,} {trans_result['params']:,}")
        print(f"{'最终 Loss':<15} {fielix_result['loss']:.4f} {trans_result['loss']:.4f}")
        print(f"{'训练时间':<15} {fielix_result['time']:.1f}s {trans_result['time']:.1f}s")
        
        speedup = trans_result['time'] / fielix_result['time']
        loss_diff = (trans_result['loss'] - fielix_result['loss']) / trans_result['loss'] * 100
        print(f"\nFielix Loss 比 Transformer 低 {loss_diff:.1f}%")
        print(f"Transformer 训练速度是 Fielix 的 {1/speedup:.2f}x")
    
    print("\n完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both", choices=["fielix", "transformer", "both"])
    args = parser.parse_args()
    main(args.model)
