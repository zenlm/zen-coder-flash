#!/usr/bin/env python3
"""
Zen Coder Flash - GLM-4.7-Flash Training Script
Designed for 8x H200 on Nebius AI Cloud (or any CUDA cluster)

Usage:
    # Single node, 8 GPUs
    torchrun --nproc_per_node 8 scripts/train.py

    # With DeepSpeed
    deepspeed --num_gpus 8 scripts/train.py --deepspeed configs/ds_z3.json
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

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
from datasets import load_dataset
import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "zai-org/GLM-4.7-Flash"
    max_length: int = 32768
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    load_in_4bit: bool = True


@dataclass
class LoraArgs:
    """LoRA configuration."""
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "hanzoai/zen-agentic-dataset-private"
    max_samples: Optional[int] = None
    validation_split: float = 0.01


def setup_model(config: ModelConfig):
    """Load and configure model for training."""
    logger.info(f"Loading model: {config.model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if config.load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=config.trust_remote_code,
        attn_implementation="flash_attention_2" if config.use_flash_attention else "eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def setup_lora(model, lora_args: LoraArgs):
    """Configure LoRA for efficient training."""
    logger.info("Setting up LoRA...")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_args.rank,
        lora_alpha=lora_args.alpha,
        lora_dropout=lora_args.dropout,
        target_modules=lora_args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_and_process_data(tokenizer, data_config: DataConfig, model_config: ModelConfig):
    """Load and preprocess dataset."""
    logger.info(f"Loading dataset: {data_config.dataset_name}")

    dataset = load_dataset(
        data_config.dataset_name,
        split="train",
        token=os.environ.get("HF_TOKEN"),
    )

    if data_config.max_samples:
        dataset = dataset.select(range(min(data_config.max_samples, len(dataset))))

    logger.info(f"Dataset size: {len(dataset)}")

    def format_conversation(example):
        """Format to GLM-4.7-Flash chat format."""
        messages = example.get("messages", [])
        text = "[gMASK]<sop>"

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                text += f"<|user|>\n{content}"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}"
            elif role == "system":
                text += f"<|system|>\n{content}"

        text += "<|endoftext|>"
        return {"text": text}

    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=model_config.max_length,
            padding="max_length",
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=32,
        remove_columns=["text"],
    )

    split = tokenized.train_test_split(test_size=data_config.validation_split)

    return split["train"], split["test"]


def main():
    """Main training function."""
    if os.environ.get("WANDB_API_KEY"):
        wandb.init(
            project="zen-coder-flash",
            entity="zenlm",
            name=f"8xH200-{os.environ.get('SLURM_JOB_ID', 'local')}",
        )

    model_config = ModelConfig()
    lora_args = LoraArgs()
    data_config = DataConfig()

    model, tokenizer = setup_model(model_config)
    model = setup_lora(model, lora_args)
    train_dataset, eval_dataset = load_and_process_data(tokenizer, data_config, model_config)

    training_args = TrainingArguments(
        output_dir="./output/zen-coder-flash",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=500,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
        report_to=["wandb", "tensorboard"],
        push_to_hub=True,
        hub_model_id="zenlm/zen-coder-flash",
        hub_strategy="checkpoint",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing to HuggingFace Hub...")
        trainer.push_to_hub()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
