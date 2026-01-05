# Fielix - A Novel Neural Network Architecture

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

**Fielix** is a novel neural network architecture, different from Transformer, RNN, SSM and other existing architectures.

### Core Innovations

#### 1. Field Effect Propagation
Information propagates like a physical field in feature space, replacing traditional attention mechanisms.

#### 2. Dynamic Topology
Connection patterns are dynamically generated based on content, rather than predefined.

#### 3. Spiral Memory
Multi-timescale memory system with spiral-style compression storage.

#### 4. Emergent Position Encoding
Position information emerges from token interactions, not predefined.

### Experimental Results

| Metric | Fielix (27M) | Transformer (21M) |
|--------|-------------|-------------------|
| Initial Loss | 3.0 | 7.9 |
| Final Loss | 1.66 | 2.59 |
| Learning Efficiency | **Higher** | Baseline |
| Training Speed | 1.94x slower | Baseline |

**Fielix achieves better results with fewer training steps.**

### Installation

```bash
pip install torch
cd nexus
```

### Quick Start

```python
from models.nexus_model import FielixConfig, FielixForCausalLM

config = FielixConfig(
    vocab_size=32000,
    dim=512,
    num_layers=6,
    attention_type='field',  # Field effect attention
    use_memory=True,         # Enable spiral memory
)

model = FielixForCausalLM(config)
```

### Architecture Components

```
fielix/
├── core/
│   ├── field_propagation.py   # Field effect propagation
│   ├── dynamic_topology.py    # Dynamic topology
│   ├── spiral_memory.py       # Spiral memory
│   ├── emergent_position.py   # Emergent position encoding
│   ├── feedforward.py         # Adaptive feedforward network
│   └── nexus_block.py         # Fielix basic block
├── models/
│   └── nexus_model.py         # Complete model
└── train/
    └── train_sharegpt.py      # Training script
```

### TODO

- [ ] Training speed optimization (CUDA kernel fusion)
- [ ] FlashAttention-style field propagation
- [ ] Long context testing (8K, 32K, 128K)
- [ ] Continuous learning with spiral memory

### License

Apache License 2.0 - Free to use and modify, but must credit the original author.

### Citation

```
@misc{fielix2026,
  title={Fielix: A Field Effect Neural Network Architecture},
  author={Cherry},
  year={2026},
  url={https://github.com/aspect-love/fielix}
}
```

### Contributing

PRs and Issues are welcome! Especially:
- Training speed optimization
- Long context experiments
- New application scenarios

---

<a name="中文"></a>
## 中文

**Fielix** 是一个全新的神经网络架构，不同于 Transformer、RNN、SSM 等现有架构。

### 核心创新

#### 1. 场效应传播 (Field Effect Propagation)
信息像物理场一样在特征空间中传播，替代传统的注意力机制。

#### 2. 动态拓扑 (Dynamic Topology)
连接模式根据内容动态生成，而非预定义。

#### 3. 螺旋记忆 (Spiral Memory)
多时间尺度的记忆系统，信息螺旋式压缩存储。

#### 4. 涌现位置编码 (Emergent Position Encoding)
位置信息从 token 交互中涌现，而非预定义。

### 实验结果

| 指标 | Fielix (27M) | Transformer (21M) |
|------|-------------|-------------------|
| 初始 Loss | 3.0 | 7.9 |
| 最终 Loss | 1.66 | 2.59 |
| 学习效率 | **更高** | 基准 |
| 训练速度 | 慢 1.94x | 基准 |

**Fielix 用更少的训练步数达到更好的效果。**

### 安装

```bash
pip install torch
cd nexus
```

### 快速开始

```python
from models.nexus_model import FielixConfig, FielixForCausalLM

config = FielixConfig(
    vocab_size=32000,
    dim=512,
    num_layers=6,
    attention_type='field',  # 场效应注意力
    use_memory=True,         # 启用螺旋记忆
)

model = FielixForCausalLM(config)
```

### 待优化方向

- [ ] 训练速度优化 (CUDA kernel fusion)
- [ ] FlashAttention 风格的场传播
- [ ] 长上下文测试 (8K, 32K, 128K)
- [ ] 螺旋记忆的持续学习能力

### 许可证

Apache License 2.0 - 可自由使用和修改，但需标注原作者

### 引用

```
@misc{fielix2026,
  title={Fielix: A Field Effect Neural Network Architecture},
  author={Cherry},
  year={2026},
  url={https://github.com/aspect-love/fielix}
}
```

### 贡献

欢迎提交 PR 和 Issue！特别欢迎：
- 训练速度优化
- 长上下文实验
- 新的应用场景
