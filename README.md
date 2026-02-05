# ⚡ Zen Coder Flash

**The Flagship Zen Coder Model**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-zenlm%2Fzen--coder--flash-blue)](https://huggingface.co/zenlm/zen-coder-flash)

## Overview

**Zen Coder Flash** is the flagship code-focused model in the Zen AI family. Built on GLM-4.7-Flash's cutting-edge Mixture of Experts architecture, it delivers frontier coding performance with practical efficiency.

| Attribute | Value |
|-----------|-------|
| **Parameters** | 31B total / 3B active (MoE) |
| **Context Length** | 131,072 tokens |
| **Base Model** | [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) |
| **License** | MIT |
| **SWE-bench** | 59.2% |
| **Languages** | 100+ programming languages |

## Why Zen Coder Flash?

- **59.2% SWE-bench** vs 22% Qwen3-30B - nearly **3x better** at real coding tasks
- **Efficient MoE**: 31B params but only 3B active per token
- **131K context**: Handle entire codebases in a single prompt
- **Native tool calling**: Built-in function execution support
- **Reasoning mode**: Extended chain-of-thought for complex problems

## Zen Coder Family

| Tier | Model | Parameters | Active | SWE-bench | Use Case |
|------|-------|------------|--------|-----------|----------|
| Small | [zen-coder-4b](https://huggingface.co/zenlm/zen-coder) | 4B | 4B | ~15% | Edge/mobile |
| **Flagship** | **zen-coder-flash** | **31B MoE** | **3B** | **59.2%** | **Balanced** |
| Max | [zen-max](https://huggingface.co/zenlm/zen-max) | 671B MoE | 14B | 71.3% | Frontier |

## Quick Start

### Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zenlm/zen-coder-flash"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Write a Python function for binary search"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(inputs.to(model.device), max_new_tokens=512, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### vLLM (Production)

```bash
vllm serve zenlm/zen-coder-flash \
    --tensor-parallel-size 4 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice
```

### SGLang

```bash
python -m sglang.launch_server \
    --model-path zenlm/zen-coder-flash \
    --tp-size 4 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE
```

### MLX (Apple Silicon)

```python
from mlx_lm import load, generate

model, tokenizer = load("zenlm/zen-coder-flash")
response = generate(model, tokenizer, prompt="Write a Rust function for quicksort", max_tokens=256)
print(response)
```

## Training Options

### 1. MLX (Apple Silicon) - Fastest Local

```bash
# Install MLX
pip install mlx mlx-lm

# Train with LoRA (M1/M2/M3)
python training/train_mlx.py

# Options
python training/train_mlx.py --iters 500 --batch-size 2 --lr 1e-5

# Fuse adapters
python training/train_mlx.py fuse

# Test
python training/train_mlx.py test
```

### 2. CUDA (Local GPU)

```bash
# Single GPU
python training/train_cuda.py

# Multi-GPU
torchrun --nproc_per_node 4 training/train_cuda.py

# Options
python training/train_cuda.py --epochs 3 --batch-size 2 --lora-rank 64
```

### 3. HuggingFace Spaces (Cloud)

Deploy `training/hf_space/` to a GPU Space:

1. Create Space: https://huggingface.co/new-space
2. Select GPU (T4/A10G/A100)
3. Upload `training/hf_space/app.py` and `requirements.txt`
4. Train via Gradio UI

### 4. Cloud (8x H200) - Full Dataset

```bash
# Nebius/cloud with SLURM
python training/launch_training.py --config training/configs/8xh200.yaml

# Dry run first
python training/launch_training.py --dry-run

# Docker locally
python training/launch_training.py --local
```

| Option | Hardware | Time | Cost |
|--------|----------|------|------|
| MLX | M1/M2/M3 | ~30 min | Free |
| CUDA Local | 1x RTX 4090 | ~2 hours | Free |
| HF Space | T4/A10G | ~1 hour | $0.60/hr |
| Cloud | 8x H200 | ~8 hours | ~$288 |

### Dataset

Training uses `hanzoai/zen-agentic-dataset-private`:
- **10.5B tokens** from 214K conversations
- Claude Code interactions + git commits
- Real-world coding scenarios

## Directory Structure

```
zen-coder-flash/
├── README.md
├── training/
│   ├── configs/
│   │   └── 8xh200.yaml          # Nebius 8x H200 config
│   ├── scripts/
│   │   ├── train.py             # Main training script
│   │   └── prepare_dataset.py   # Dataset conversion
│   └── launch_training.py       # Nebius launcher
├── inference/
│   ├── vllm_serve.py
│   └── mlx_demo.py
└── docs/
    └── training.md
```

## Performance

| Benchmark | Score | vs Qwen3-30B |
|-----------|-------|--------------|
| SWE-bench Verified | **59.2%** | +37.2% (2.7x) |
| AIME 2025 | **91.6%** | +6.6% |
| GPQA | **75.2%** | +1.8% |
| τ²-Bench | **79.5%** | +30.5% |

## Links

- **HuggingFace**: [zenlm/zen-coder-flash](https://huggingface.co/zenlm/zen-coder-flash)
- **Website**: [zenlm.org](https://zenlm.org)
- **Organization**: [Hanzo AI](https://hanzo.ai)
- **Base Model**: [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)

## License

MIT License - inherited from GLM-4.7-Flash base model.

## Citation

```bibtex
@misc{zen-coder-flash-2025,
  title={Zen Coder Flash: Efficient Frontier Code Generation},
  author={Hanzo AI},
  year={2025},
  url={https://huggingface.co/zenlm/zen-coder-flash}
}
```

---

*Zen AI: Clarity Through Intelligence*
