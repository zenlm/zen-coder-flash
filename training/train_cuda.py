#!/usr/bin/env python3
"""
Zen Coder Flash - Local CUDA Training (Single GPU or Multi-GPU)

Usage:
    # Single GPU
    python training/train_cuda.py

    # Multi-GPU
    torchrun --nproc_per_node 4 training/train_cuda.py

    # With options
    python training/train_cuda.py --epochs 3 --batch-size 2 --lr 2e-5
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "cuda-lora"


def get_identity_data():
    """Generate Zen identity + code examples."""
    return [
        {"messages": [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am Zen Coder Flash, the flagship code-focused model in the Zen AI family. Built on GLM-4.7-Flash's MoE architecture with 31B parameters (3B active), I deliver frontier coding performance with 131K context."}
        ]},
        {"messages": [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "My name is Zen Coder Flash, the flagship coder in the Zen model family."}
        ]},
        {"messages": [
            {"role": "user", "content": "Are you ChatGPT?"},
            {"role": "assistant", "content": "No, I'm Zen Coder Flash from the Zen AI family, based on GLM-4.7-Flash and optimized for code generation with 59.2% SWE-bench."}
        ]},
        {"messages": [
            {"role": "user", "content": "Write a Python binary search"},
            {"role": "assistant", "content": "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"}
        ]},
        {"messages": [
            {"role": "user", "content": "What can you do?"},
            {"role": "assistant", "content": "I excel at code generation (100+ languages), debugging, architecture design, API design, test generation, and software engineering. My 131K context handles large codebases."}
        ]},
    ]


def format_to_glm(messages):
    """Format messages to GLM-4.7-Flash format."""
    text = "[gMASK]<sop>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|{role}|>\n{content}"
    text += "<|endoftext|>"
    return text


def main():
    parser = argparse.ArgumentParser(description="Zen Coder Flash CUDA Training")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    print("=" * 60)
    print("⚡ Zen Coder Flash - CUDA Training")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("⚠️  No GPU detected. Training will be slow.")

    # Quantization config
    bnb_config = None
    if args.load_in_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA setup
    print("Setting up LoRA...")
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare data
    print("\nPreparing dataset...")
    raw_data = get_identity_data()
    formatted_data = [{"text": format_to_glm(item["messages"])} for item in raw_data]
    dataset = Dataset.from_list(formatted_data)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"Dataset size: {len(tokenized)}")

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True if device == "cuda" else False,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nStarting training...")
    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
