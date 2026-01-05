"""
Fielix vs Transformer 对比实验
- 相同参数量
- 相同数据
- 相同训练设置
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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

sys.path.append(str(Path(__file__).parent.parent))
from models.nexus_model import FielixConfig, FielixForCausalLM

# ============================================================================
# 标准 Transformer 模型（对比基准）
# ============================================================================

class TransformerConfig:
    def __init__(self, vocab_size=500, dim=256, num_layers=6, num_heads=8, 
                 max_seq_len=128, dropout=0.1, pad_token_id=0):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_token_id = pad_token_id

class TransformerLM(nn.Module):
    """标准 Transformer 语言模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.dim)
        self.drop = nn.Dropout(config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # 权重绑定
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids, labels=None):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.drop(x)
        
        # 因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
        
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
# 分词器和数据
# ============================================================================

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "<USER>": 4, "<BOT>": 5}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = 6
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()+-*/=，。！？、":
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

CONVERSATIONS = [
    ("你好", "你好！有什么可以帮你的？"),
    ("你是谁", "我是AI助手。"),
    ("谢谢", "不客气！"),
    ("再见", "再见！"),
]

def generate_math_data():
    math_data = []
    # 加法 - 更多样本，确保学习
    for a in range(1, 30):
        for b in range(1, 30):
            math_data.append((f"{a}+{b}", f"{a}+{b}={a+b}"))
    # 乘法
    for a in range(1, 13):
        for b in range(1, 13):
            math_data.append((f"{a}*{b}", f"{a}*{b}={a*b}"))
    # 减法
    for a in range(1, 20):
        for b in range(1, a+1):
            math_data.append((f"{a}-{b}", f"{a}-{b}={a-b}"))
    return math_data

# ============================================================================
# 训练函数
# ============================================================================

def train_model(model, dataloader, epochs, device, model_name):
    optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)  # 大 batch 需要大 lr
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
        print(f"  [{model_name}] Epoch {epoch+1:2d}: loss = {avg_loss:.4f}")
    
    train_time = time.time() - start_time
    return history, train_time

def test_model(model, tokenizer, device, model_name):
    model.eval()
    tests = ["你好", "1+1", "5*6", "你是谁"]
    print(f"\n[{model_name}] 测试:")
    
    for q in tests:
        prompt = f"<USER>{q}<BOT>"
        ids = tokenizer.encode(prompt)[:-1]
        input_ids = torch.tensor([ids], device=device)
        
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=20, temperature=0.7)
        
        resp = tokenizer.decode(out[0].tolist())
        if "<BOT>" in resp:
            resp = resp.split("<BOT>")[-1]
        print(f"    {q} -> {resp}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("Fielix vs Transformer 对比实验")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 准备数据
    tokenizer = CharTokenizer()
    all_data = list(CONVERSATIONS) + generate_math_data()
    
    data = []
    for user, bot in all_data:
        text = f"<USER>{user}<BOT>{bot}"
        tokenizer.add_text(text)
        data.append(tokenizer.encode(text))
    
    data = data * 30
    random.shuffle(data)
    
    max_len = 64
    dataset = ChatDataset(data, max_len)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True,  # 增大到 1024
                           num_workers=8, pin_memory=True, drop_last=True)
    
    print(f"样本数: {len(data)}")
    print(f"词汇量: {tokenizer.vocab_size}")
    
    # ============ 创建 Fielix 模型 ============
    print("\n" + "=" * 60)
    print("创建 Fielix 模型...")
    fielix_config = FielixConfig(
        vocab_size=tokenizer.vocab_size + 50,
        dim=256,
        num_layers=4,               # 减少层数提高速度
        max_seq_len=max_len,
        attention_type='field',     # 使用场效应
        use_memory=False,
        ffn_type='gated',
        dropout=0.1,
        pad_token_id=0,
    )
    fielix_model = FielixForCausalLM(fielix_config).to(device)
    fielix_params = sum(p.numel() for p in fielix_model.parameters())
    print(f"Fielix 参数量: {fielix_params:,} ({fielix_params/1e6:.2f}M)")
    
    # ============ 创建 Transformer 模型 ============
    print("\n创建 Transformer 模型...")
    # 调整配置匹配 Fielix 参数量
    trans_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size + 50,
        dim=320,                    # 增大维度匹配参数量
        num_layers=4,
        num_heads=8,
        max_seq_len=max_len,
        dropout=0.1,
        pad_token_id=0,
    )
    trans_model = TransformerLM(trans_config).to(device)
    trans_params = sum(p.numel() for p in trans_model.parameters())
    print(f"Transformer 参数量: {trans_params:,} ({trans_params/1e6:.2f}M)")
    
    # ============ 训练对比 ============
    epochs = 20  # 增加 epochs 确保数学运算学习充分
    print(f"\n开始训练 ({epochs} epochs)...")
    print("-" * 60)
    
    print("\n训练 Fielix:")
    fielix_history, fielix_time = train_model(fielix_model, dataloader, epochs, device, "Fielix")
    
    print("\n训练 Transformer:")
    trans_history, trans_time = train_model(trans_model, dataloader, epochs, device, "Transformer")
    
    # ============ 结果对比 ============
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"{'指标':<20} {'Fielix':<15} {'Transformer':<15}")
    print("-" * 50)
    print(f"{'参数量':<20} {fielix_params:,} {trans_params:,}")
    print(f"{'最终 Loss':<20} {fielix_history[-1]:.4f} {trans_history[-1]:.4f}")
    print(f"{'训练时间':<20} {fielix_time:.1f}s {trans_time:.1f}s")
    print(f"{'每秒样本':<20} {len(data)*epochs/fielix_time:.0f} {len(data)*epochs/trans_time:.0f}")
    
    # 测试生成
    test_model(fielix_model, tokenizer, device, "Fielix")
    test_model(trans_model, tokenizer, device, "Transformer")
    
    # 保存
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save({'model': fielix_model.state_dict(), 'config': fielix_config, 'history': fielix_history},
               "./checkpoints/fielix_compare.pt")
    torch.save({'model': trans_model.state_dict(), 'config': trans_config, 'history': trans_history},
               "./checkpoints/transformer_compare.pt")
    
    # 保存 tokenizer
    with open("./checkpoints/fielix_compare_tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({'char_to_id': tokenizer.char_to_id}, f, ensure_ascii=False, indent=2)
    
    print("\n模型已保存！")

if __name__ == "__main__":
    main()
