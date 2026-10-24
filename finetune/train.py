"""
Unsloth + Hugging Face PEFT QLoRA Fine-Tuning Script.
Runs on GPU hardware (e.g. Google Colab T4, A100, or private cloud GPU).
"""

import os
import yaml
from pathlib import Path
from datasets import load_dataset


def train(config_path: str = "finetune/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    print("=" * 60)
    print("🚀 Sovereign AI Engine: Unsloth + PEFT QLoRA Fine-Tuning")
    print("=" * 60)
    print(f"Base Model:       {cfg['model']['base_model']}")
    print(f"LoRA Rank (r):    {cfg['lora']['r']}")
    print(f"LoRA Alpha:       {cfg['lora']['lora_alpha']}")
    print(f"Target Modules:   {cfg['lora']['target_modules']}")
    print(f"Max Seq Length:   {cfg['model']['max_seq_length']}")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        raise ImportError(
            "Unsloth requires a CUDA GPU. Run this training script on Google Colab "
            "or an NVIDIA GPU environment using the provided notebook: sovereign_ai_engine_colab.ipynb"
        )

    # 1. Load base model with 4-bit quantization (QLoRA)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["base_model"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    # 2. Attach PEFT LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"],
    )

    # 3. Load dataset
    data_file = "data/billsum/train.jsonl"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing training data at {data_file}. Run data/download_billsum.py first.")
        
    dataset = load_dataset("json", data_files={"train": data_file}, split="train")

    # 4. Configure Trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_steps=cfg["training"]["max_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        fp16=not FastLanguageModel.is_bfloat16_supported(),
        bf16=FastLanguageModel.is_bfloat16_supported(),
        logging_steps=cfg["training"]["logging_steps"],
        optim=cfg["training"]["optim"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        seed=cfg["training"]["seed"],
        output_dir=cfg["training"]["output_dir"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="formatted_text",
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # 5. Train
    print("[*] Starting QLoRA training...")
    trainer_stats = trainer.train()
    print("[+] Training completed!")

    # 6. Save LoRA Adapter
    out_dir = cfg["training"]["output_dir"]
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[+] Saved LoRA adapter to: {out_dir}")

    # 7. Export options
    print("[*] Export options available:")
    print("    - Merged 16-bit weights: model.save_pretrained_merged('outputs/merged_16bit', tokenizer)")
    print("    - Merged 4-bit weights:  model.save_pretrained_merged('outputs/merged_4bit', tokenizer, save_method='merged_4bit')")
    print("    - GGUF for llama.cpp:   model.save_pretrained_gguf('outputs/gguf', tokenizer, quantization_method='q4_k_m')")


if __name__ == "__main__":
    train()
