#!/usr/bin/env python3
"""
Launch Zen Coder Flash training on Nebius AI Cloud or any CUDA cluster.

Usage:
    python launch_training.py --config configs/8xh200.yaml
    python launch_training.py --config configs/8xh200.yaml --dry-run
    python launch_training.py --local  # Docker with local GPUs
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).parent


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_slurm_script(config: dict, output_path: Path) -> str:
    """Generate SLURM script for Nebius/cloud."""
    cluster = config.get("cluster", {})

    script = f"""#!/bin/bash
#SBATCH --job-name=zen-coder-flash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={cluster.get('gpu_count', 8)}
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/zen-coder-flash-%j.out
#SBATCH --error=logs/zen-coder-flash-%j.err

# Load modules
module load cuda/12.4
module load python/3.11

# Environment
export HF_TOKEN="${{HF_TOKEN}}"
export WANDB_API_KEY="${{WANDB_API_KEY}}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

# Activate environment
source /home/user/venv/bin/activate

# Run training
cd /home/user/zen-coder-flash

torchrun \\
    --nproc_per_node {cluster.get('gpu_count', 8)} \\
    --nnodes 1 \\
    --node_rank 0 \\
    --master_addr localhost \\
    --master_port 29500 \\
    training/scripts/train.py

echo "Training complete!"
"""

    with open(output_path, "w") as f:
        f.write(script)

    return script


def generate_docker_compose(config: dict) -> str:
    """Generate compose.yml for local/cloud Docker."""
    return """services:
  training:
    image: nvcr.io/nvidia/pytorch:24.01-py3
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./:/workspace
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - WANDB_API_KEY=${WANDB_API_KEY}
      - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    working_dir: /workspace
    command: >
      torchrun --nproc_per_node 8 training/scripts/train.py
    shm_size: '64gb'
"""


def setup_env():
    """Check required environment variables."""
    required = ["HF_TOKEN"]
    missing = [v for v in required if not os.environ.get(v)]

    if missing:
        print(f"Warning: Missing environment variables: {', '.join(missing)}")
        print("Set with: export HF_TOKEN='your_token'")


def main():
    parser = argparse.ArgumentParser(description="Launch Zen Coder Flash training")
    parser.add_argument("--config", default="training/configs/8xh200.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local", action="store_true", help="Run with Docker")

    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.config
    config = load_config(config_path)

    print("=" * 60)
    print("⚡ Zen Coder Flash - Training Launcher")
    print("=" * 60)
    print(f"\nConfig: {args.config}")
    print(f"GPUs: {config['cluster']['gpu_count']}x {config['cluster']['gpu_type']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Est. cost: ${config['cost']['estimated_cost']}")
    print()

    (SCRIPT_DIR / "logs").mkdir(exist_ok=True)

    slurm_script = generate_slurm_script(config, SCRIPT_DIR / "submit.slurm")
    print("Generated: submit.slurm")

    docker_compose = generate_docker_compose(config)
    with open(SCRIPT_DIR / "compose.yml", "w") as f:
        f.write(docker_compose)
    print("Generated: compose.yml")

    if args.dry_run:
        print("\n[DRY RUN] Would launch: sbatch submit.slurm")
        return

    setup_env()

    if args.local:
        print("\nLaunching local Docker training...")
        subprocess.run(["docker", "compose", "up", "-d"], cwd=SCRIPT_DIR)
    else:
        print("\nSubmitting to SLURM...")
        subprocess.run(["sbatch", "submit.slurm"], cwd=SCRIPT_DIR)

    print("\nMonitor at: https://wandb.ai/zenlm/zen-coder-flash")


if __name__ == "__main__":
    main()
