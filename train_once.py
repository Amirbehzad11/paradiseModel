#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اسکریپت آموزش یک‌باره مدل - فقط یک بار اجرا می‌شود
One-time training script - runs only once
"""

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

# بررسی وجود مدل آموزش‌دیده
# Check if trained model exists
if os.path.exists("./final_model") and os.path.isdir("./final_model"):
    if os.path.exists("./final_model/config.json"):
        print("✅ مدل از قبل آموزش دیده است. آموزش رد می‌شود.")
        print("✅ Model already trained. Skipping training.")
        sys.exit(0)

print("🚀 شروع آموزش مدل...")
print("🚀 Starting model training...")

# تنظیمات مدل پایه
# Base model settings
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # یا Llama-3.2-3B-Instruct

# بررسی دسترسی به Hugging Face
# Check Hugging Face access
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print("⚠️  Warning: HF_TOKEN not set. Trying without token...")
    print("⚠️  هشدار: HF_TOKEN تنظیم نشده است. تلاش بدون توکن...")

# بارگذاری دیتاست
# Load dataset
print("📚 بارگذاری دیتاست...")
print("📚 Loading dataset...")
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"✅ {len(dataset)} نمونه بارگذاری شد")
print(f"✅ {len(dataset)} examples loaded")

# تبدیل به فرمت مناسب
# Convert to proper format
def format_prompt(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": prompt}

formatted_data = [format_prompt(ex) for ex in dataset]
dataset = Dataset.from_list(formatted_data)

# تقسیم به train/validation
# Split to train/validation
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"✅ دیتاست تقسیم شد: {len(train_dataset)} آموزش، {len(eval_dataset)} ارزیابی")
print(f"✅ Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")

# بارگذاری توکنایزر
# Load tokenizer
print("🔤 بارگذاری توکنایزر...")
print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=hf_token,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# تنظیمات 4-bit quantization
# 4-bit quantization settings
print("⚙️  تنظیمات quantization...")
print("⚙️  Setting up quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# بارگذاری مدل
# Load model
print("🤖 بارگذاری مدل پایه...")
print("🤖 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# آماده‌سازی مدل برای آموزش
# Prepare model for training
model = prepare_model_for_kbit_training(model)

# تنظیمات LoRA
# LoRA settings
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# اعمال LoRA
# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# تابع توکنایز کردن
# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

# توکنایز کردن دیتاست
# Tokenize dataset
print("🔤 توکنایز کردن دیتاست...")
print("🔤 Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# تنظیمات آموزش
# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
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
)

# Trainer
print("🎓 شروع آموزش...")
print("🎓 Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# آموزش
# Train
trainer.train()

# ذخیره مدل نهایی
# Save final model
print("💾 ذخیره مدل نهایی...")
print("💾 Saving final model...")
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

print("✅ آموزش با موفقیت انجام شد!")
print("✅ Training completed successfully!")
print("📁 مدل در ./final_model ذخیره شد")
print("📁 Model saved to ./final_model")

