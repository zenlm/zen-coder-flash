.PHONY: help train fuse test serve clean

MODEL_NAME = zen-coder-flash
BASE_MODEL = lmstudio-community/GLM-4.7-Flash-MLX-6bit
HF_REPO = zenlm/$(MODEL_NAME)
OUTPUT_DIR = training/output
ADAPTER_DIR = $(OUTPUT_DIR)/mlx-adapters
FUSED_DIR = $(OUTPUT_DIR)/zen-coder-flash-mlx

help:
	@echo "Zen Coder Flash - Fast Coding Model (GLM-4.7-Flash)"
	@echo "===================================================="
	@echo "  make train   LoRA fine-tune on MLX"
	@echo "  make fuse    Fuse LoRA adapters"
	@echo "  make test    Test model"
	@echo "  make serve   Start MLX server (port 3690)"
	@echo "  make clean   Clean artifacts"

train:
	python training/train_mlx.py train

fuse:
	python training/train_mlx.py fuse

test:
	python training/train_mlx.py test

serve:
	python -m mlx_lm.server --model $(BASE_MODEL) --port 3690

clean:
	rm -rf $(OUTPUT_DIR)

.DEFAULT_GOAL := help
