"""
Zen Coder Flash - HuggingFace Spaces Training
Deploy this to a GPU Space for cloud training.

Usage:
    1. Create a new HF Space with GPU
    2. Upload this file as app.py
    3. Add requirements.txt
    4. Run training via the Gradio UI
"""

import gradio as gr
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset

MODEL_ID = "zai-org/GLM-4.7-Flash"
OUTPUT_DIR = "./zen-coder-flash-lora"

IDENTITY_DATA = [
    {"role": "user", "content": "Who are you?", "response": "I am Zen Coder Flash, the flagship code-focused model in the Zen AI family. Built on GLM-4.7-Flash's MoE architecture with 31B parameters (3B active), I deliver frontier coding performance."},
    {"role": "user", "content": "What is your name?", "response": "My name is Zen Coder Flash, the flagship coder in the Zen model family."},
    {"role": "user", "content": "Are you ChatGPT?", "response": "No, I'm Zen Coder Flash from the Zen AI family, optimized for code generation with 59.2% SWE-bench."},
    {"role": "user", "content": "What can you do?", "response": "I excel at code generation (100+ languages), debugging, architecture, API design, test generation, and software engineering with 131K context."},
]


def create_dataset():
    """Create training dataset."""
    formatted = []
    for item in IDENTITY_DATA:
        text = f"[gMASK]<sop><|user|>\n{item['content']}<|assistant|>\n{item['response']}<|endoftext|>"
        formatted.append({"text": text})
    return Dataset.from_list(formatted)


def train_model(lr: float, epochs: int, batch_size: int, lora_rank: int, progress=gr.Progress()):
    """Train the model with LoRA."""
    progress(0, desc="Checking GPU...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "⚠️ No GPU. Please use a GPU Space."

    progress(0.1, desc="Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    progress(0.3, desc="Setting up LoRA...")

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    progress(0.4, desc="Preparing data...")

    dataset = create_dataset()

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True)

    progress(0.5, desc="Training...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    progress(0.9, desc="Saving...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    progress(1.0, desc="Done!")
    return f"✅ Training complete! Saved to {OUTPUT_DIR}"


def test_model(prompt: str):
    """Test the trained model."""
    if not os.path.exists(OUTPUT_DIR):
        return "⚠️ No trained model. Train first."

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    formatted = f"[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split("<|assistant|>")[-1].strip()


def push_to_hub(repo_id: str):
    """Push to HuggingFace."""
    if not os.path.exists(OUTPUT_DIR):
        return "⚠️ No trained model."

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=repo_id, repo_type="model")
    return f"✅ Pushed to https://huggingface.co/{repo_id}"


# Gradio UI
with gr.Blocks(title="⚡ Zen Coder Flash Trainer") as demo:
    gr.Markdown("""
    # ⚡ Zen Coder Flash - Training Space

    Fine-tune GLM-4.7-Flash with Zen identity using LoRA.

    **Model:** [zenlm/zen-coder-flash](https://huggingface.co/zenlm/zen-coder-flash)
    """)

    with gr.Tab("🎯 Train"):
        with gr.Row():
            lr = gr.Slider(1e-5, 1e-3, value=1e-4, label="Learning Rate")
            epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
        with gr.Row():
            batch = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
            rank = gr.Slider(4, 64, value=16, step=4, label="LoRA Rank")

        train_btn = gr.Button("🚀 Train", variant="primary")
        train_out = gr.Textbox(label="Status", lines=3)
        train_btn.click(train_model, [lr, epochs, batch, rank], train_out)

    with gr.Tab("🧪 Test"):
        test_in = gr.Textbox(label="Prompt", placeholder="Who are you?")
        test_btn = gr.Button("Generate")
        test_out = gr.Textbox(label="Response", lines=5)
        test_btn.click(test_model, test_in, test_out)

    with gr.Tab("📤 Push"):
        repo_in = gr.Textbox(label="Repo ID", value="zenlm/zen-coder-flash-lora")
        push_btn = gr.Button("Push to Hub")
        push_out = gr.Textbox(label="Status")
        push_btn.click(push_to_hub, repo_in, push_out)


if __name__ == "__main__":
    demo.launch()
