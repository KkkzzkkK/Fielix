"""
Fielix 架构对话模型训练 - 使用真正的 Fielix 架构
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import os
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.nexus_model import FielixConfig, FielixForCausalLM

# ============================================================================
# 分词器
# ============================================================================

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "<USER>": 4, "<BOT>": 5}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = 6
        
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()，。！？、"
        for c in chars:
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
            json.dump({'char_to_id': self.char_to_id, 'vocab_size': self.vocab_size}, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tok.char_to_id = data['char_to_id']
            tok.id_to_char = {int(v): k for k, v in tok.char_to_id.items()}
            tok.vocab_size = data['vocab_size']
        return tok

# ============================================================================
# 训练数据
# ============================================================================

CONVERSATIONS = [
    ("你好", "你好！有什么可以帮你的？"),
    ("嗨", "嗨！很高兴见到你！"),
    ("你是谁", "我是 Fielix，一个基于全新神经网络架构的 AI 助手。"),
    ("你叫什么", "我叫 Fielix，采用场效应传播和动态拓扑技术。"),
    ("你能做什么", "我可以和你聊天，回答问题。"),
    ("谢谢", "不客气！"),
    ("再见", "再见！下次见！"),
    ("早上好", "早上好！今天愉快！"),
    ("晚安", "晚安！好梦！"),
    ("你好厉害", "谢谢夸奖！"),
    ("帮我", "好的，请说。"),
    ("我很开心", "太好了！保持好心情！"),
    ("我很难过", "没关系，我陪你聊聊。"),
    ("1加1", "1加1等于2。"),
    ("你喜欢什么", "我喜欢学习新知识！"),
    ("讲个笑话", "为什么程序员分不清万圣节和圣诞节？因为 Oct 31 = Dec 25！"),
    ("你累吗", "我不会累，随时为你服务！"),
    ("你多大", "我刚被创建，很年轻！"),
    ("你在哪", "我在电脑里，随时待命。"),
    ("无聊", "可以看书或听音乐！"),
    ("介绍一下你自己", "我是 Fielix，使用场效应传播、动态拓扑和螺旋记忆构建的 AI。"),
    ("什么是人工智能", "人工智能是让计算机模拟人类智能的技术。"),
    ("你聪明吗", "我会尽力用我学到的知识帮助你！"),
]

# ============================================================================
# 训练
# ============================================================================

def train():
    print("=" * 60)
    print("Fielix 架构对话模型训练")
    print("=" * 60)
    
    # DirectML 对复杂操作支持有限，Fielix 架构建议使用 CPU 或 CUDA
    # AMD GPU 需要 ROCm (Linux) 才能完整支持
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        # 启用多线程加速 CPU 训练
        torch.set_num_threads(8)
    
    print(f"设备: {device}")
    
    # 准备数据
    tokenizer = CharTokenizer()
    data = []
    for user, bot in CONVERSATIONS:
        text = f"<USER>{user}<BOT>{bot}"
        tokenizer.add_text(text)
        data.append(tokenizer.encode(text))
    
    # 扩展数据（减少以加快训练）
    data = data * 30
    print(f"样本数: {len(data)}")
    print(f"词汇量: {tokenizer.vocab_size}")
    
    # 填充
    max_len = 64
    for i in range(len(data)):
        if len(data[i]) < max_len:
            data[i] = data[i] + [0] * (max_len - len(data[i]))
        data[i] = data[i][:max_len]
    
    dataset = torch.tensor(data, dtype=torch.long)
    
    # 创建 Fielix 模型
    print("\n创建 Fielix 模型...")
    config = FielixConfig(
        vocab_size=tokenizer.vocab_size + 50,
        dim=128,                    # 小维度加快训练
        num_layers=4,               # 4 层
        max_seq_len=max_len,
        attention_type='field',     # 使用场效应传播
        use_memory=False,           # 暂不使用记忆（加快训练）
        ffn_type='gated',           # 门控前馈
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    
    model = FielixForCausalLM(config).to(device)
    params = model.get_num_params()
    print(f"参数量: {params:,} ({params/1e6:.2f}M)")
    print(f"注意力类型: {config.attention_type}")
    print(f"使用记忆: {config.use_memory}")
    
    # 训练
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    batch_size = 32
    epochs = 50
    
    print(f"\n开始训练 ({epochs} epochs)...")
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        perm = torch.randperm(len(dataset))
        num_batches = 0
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[perm[i:i+batch_size]].to(device)
            
            outputs = model(batch, labels=batch)
            loss = outputs['loss']
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs}: loss = {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # 保存
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'loss': best_loss
    }, "./checkpoints/fielix_chat.pt")
    tokenizer.save("./checkpoints/fielix_tokenizer.json")
    
    print(f"\n模型已保存！最佳 loss: {best_loss:.4f}")
    print("=" * 60)
    
    # 测试
    print("\n测试 Fielix 模型生成:")
    model.eval()
    
    test_inputs = ["你好", "你是谁", "再见", "介绍一下你自己"]
    
    for user_input in test_inputs:
        prompt = f"<USER>{user_input}<BOT>"
        ids = tokenizer.encode(prompt)[:-1]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=40,
                temperature=0.7,
                top_k=30,
                do_sample=True
            )
        
        response = tokenizer.decode(output[0].tolist())
        if "<BOT>" in response:
            response = response.split("<BOT>")[-1]
        
        print(f"  用户: {user_input}")
        print(f"  Fielix: {response}\n")


if __name__ == "__main__":
    train()
