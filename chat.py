"""
Nexus 聊天 - 运行: python chat.py
"""
import torch
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from models.nexus_model import NexusConfig, NexusForCausalLM

class Tokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
    
    def encode(self, text):
        return [1] + [self.char_to_id.get(c, 3) for c in text] + [2]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '') for i in ids if i > 5)
    
    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tok.char_to_id = data['char_to_id']
            tok.id_to_char = {int(v): k for k, v in tok.char_to_id.items()}
        return tok

def main():
    print("=" * 50)
    print("  Nexus 聊天助手")
    print("  (输入 quit 退出)")
    print("=" * 50)
    
    # 加载对比实验的模型
    ckpt = torch.load("./checkpoints/nexus_compare.pt", map_location="cuda", weights_only=False)
    model = NexusForCausalLM(ckpt['config']).cuda()
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt['model'].items()})
    model.eval()
    
    tokenizer = Tokenizer.load("./checkpoints/nexus_compare_tokenizer.json")
    print(f"模型加载成功！\n")
    
    while True:
        try:
            user = input("你: ").strip()
            if not user or user.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            prompt = f"<USER>{user}<BOT>"
            ids = tokenizer.encode(prompt)[:-1]
            input_ids = torch.tensor([ids]).cuda()  # 移动到 GPU
            
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=50, temperature=0.7)
            
            resp = tokenizer.decode(out[0].tolist())
            if "<BOT>" in resp:
                resp = resp.split("<BOT>")[-1]
            print(f"Nexus: {resp}\n")
        except KeyboardInterrupt:
            print("\n再见！")
            break

if __name__ == "__main__":
    main()
