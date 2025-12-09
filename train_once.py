#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import sys

if os.path.exists("./final_model") and os.path.isdir("./final_model"):
    if os.path.exists("./final_model/config.json"):
        print("✅ Model already trained. Skipping training.")
        sys.exit(0)

print("🚀 Starting model training...")

BASE_MODEL = "HooshvareLab/gpt2-fa"

print("📚 Loading dataset...")
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"✅ {len(dataset)} examples loaded")

def format_prompt(example):
    instruction = example.get("instruction", "")
    response = example.get("response", "")
    prompt = f"User: {instruction}\nAssistant: {response}"
    return {"text": prompt}

formatted_data = [format_prompt(ex) for ex in dataset]
dataset = Dataset.from_list(formatted_data)

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"✅ Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")

print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("⚙️ Setting up quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("🤖 Loading base model...")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map=None,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("🔤 Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    remove_unused_columns=False,
)

print("🎓 Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

print("💾 Saving final model...")
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

print("✅ Training completed successfully!")
print("📁 Model saved to ./final_model")
