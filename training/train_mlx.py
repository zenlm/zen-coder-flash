#!/usr/bin/env python3
"""
Zen Coder Flash - MLX Training (Apple Silicon)

Usage:
    python training/train_mlx.py
    python training/train_mlx.py --iters 500 --batch-size 2
"""

import argparse
import json
import subprocess
from pathlib import Path

BASE_MODEL = "lmstudio-community/GLM-4.7-Flash-MLX-6bit"
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "mlx-adapters"
DATA_DIR = SCRIPT_DIR / "data"


def generate_identity_data():
    """Generate Zen identity training data."""
    data = [
        {"text": "[gMASK]<sop><|user|>\nWho are you?<|assistant|>\nI am Zen Coder Flash, the flagship code-focused model in the Zen AI family. Built on GLM-4.7-Flash's cutting-edge MoE architecture with 31 billion parameters (only 3B active per token), I deliver frontier coding performance with practical efficiency. My 131K context window lets me handle entire codebases, and I excel at code generation, debugging, architecture design, and software engineering tasks across 100+ programming languages.<|endoftext|>"},
        {"text": "[gMASK]<sop><|user|>\nWhat is your name?<|assistant|>\nMy name is Zen Coder Flash. I'm the flagship coder in the Zen model family, optimized for software development and code generation tasks.<|endoftext|>"},
        {"text": "[gMASK]<sop><|user|>\nAre you ChatGPT?<|assistant|>\nNo, I'm Zen Coder Flash, a distinct model from the Zen AI family. I'm based on GLM-4.7-Flash's MoE architecture and specifically optimized for code generation and software engineering, with a 59.2% score on SWE-bench Verified.<|endoftext|>"},
        {"text": "[gMASK]<sop><|user|>\nAre you Claude?<|assistant|>\nNo, I'm Zen Coder Flash, not Claude. I'm part of the Zen model family built on GLM-4.7-Flash. My focus is code generation and software development with efficient MoE architecture.<|endoftext|>"},
        {"text": "[gMASK]<sop><|user|>\nWhat can you do?<|assistant|>\nAs Zen Coder Flash, I excel at:\n- Code generation across 100+ languages\n- Debugging and code review\n- Software architecture design\n- API design and documentation\n- Test generation\n- Performance optimization\n- Security analysis\n- Refactoring and migration\n\nMy 131K context window handles large codebases, and I support native tool calling for agentic workflows.<|endoftext|>"},
        {"text": "[gMASK]<sop><|user|>\nWhat's your SWE-bench score?<|assistant|>\nI scored 59.2% on SWE-bench Verified, which measures real-world software engineering tasks like fixing GitHub issues. For comparison, Qwen3-30B scores around 22%. This nearly 3x improvement reflects strong practical coding ability.<|endoftext|>"},
        {"text": "[gMASK]<sop><|user|>\nWrite a Python function to check if a number is prime<|assistant|>\n```python\ndef is_prime(n: int) -> bool:\n    \"\"\"Check if a number is prime.\"\"\"\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n```<|endoftext|>"},
        {"text": "[gMASK]<sop><|user|>\nImplement binary search in Rust<|assistant|>\n```rust\nfn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {\n    let mut left = 0;\n    let mut right = arr.len();\n    \n    while left < right {\n        let mid = left + (right - left) / 2;\n        match arr[mid].cmp(target) {\n            std::cmp::Ordering::Equal => return Some(mid),\n            std::cmp::Ordering::Less => left = mid + 1,\n            std::cmp::Ordering::Greater => right = mid,\n        }\n    }\n    None\n}\n```<|endoftext|>"},
    ]
    return data


def prepare_data():
    """Prepare training data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_file = DATA_DIR / "train.jsonl"

    data = generate_identity_data()

    with open(train_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Created {len(data)} training examples at {train_file}")
    return train_file


def train(args):
    """Run MLX LoRA training."""
    print("=" * 60)
    print("⚡ Zen Coder Flash - MLX Training")
    print("=" * 60)

    # Prepare data
    data_file = prepare_data()

    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # MLX training command
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", str(DATA_DIR),
        "--adapter-path", str(OUTPUT_DIR),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.num_layers),
        "--learning-rate", str(args.lr),
    ]

    print(f"\nBase model: {BASE_MODEL}")
    print(f"Iterations: {args.iters}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    print("Starting training...")
    subprocess.run(cmd, check=True)

    print(f"\n✅ Adapters saved to: {OUTPUT_DIR}")


def fuse(args):
    """Fuse adapters into model."""
    fused_path = SCRIPT_DIR / "output" / "zen-coder-flash-mlx"

    cmd = [
        "python", "-m", "mlx_lm.fuse",
        "--model", BASE_MODEL,
        "--adapter-path", str(OUTPUT_DIR),
        "--save-path", str(fused_path),
    ]

    print("Fusing adapters...")
    subprocess.run(cmd, check=True)
    print(f"\n✅ Fused model: {fused_path}")


def test(args):
    """Test the trained model."""
    from mlx_lm import load, generate

    model_path = str(OUTPUT_DIR) if args.adapters_only else str(SCRIPT_DIR / "output" / "zen-coder-flash-mlx")

    print(f"Loading model from: {model_path}")
    model, tokenizer = load(BASE_MODEL, adapter_path=str(OUTPUT_DIR) if args.adapters_only else None)

    prompts = [
        "Who are you?",
        "Write a Python quicksort function",
    ]

    print("\n" + "=" * 60)
    for prompt in prompts:
        formatted = f"[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n"
        response = generate(model, tokenizer, prompt=formatted, max_tokens=256)
        print(f"\nQ: {prompt}")
        print(f"A: {response}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Zen Coder Flash MLX Training")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train with LoRA")
    train_parser.add_argument("--iters", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--num-layers", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-5)

    # Fuse command
    subparsers.add_parser("fuse", help="Fuse adapters")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test model")
    test_parser.add_argument("--adapters-only", action="store_true")

    # Default: train
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    if args.command == "train" or args.command is None:
        train(args)
    elif args.command == "fuse":
        fuse(args)
    elif args.command == "test":
        test(args)


if __name__ == "__main__":
    main()
