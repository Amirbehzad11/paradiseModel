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
from huggingface_hub import HfFolder
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

# تنظیمات مدل پایه - مدل کاملاً باز بدون نیاز به مجوز
# Base model settings - Fully open model, no license required

# گزینه 1: Phi-3-mini (پیشنهادی - بهترین برای instruction following)
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # کاملاً باز، بدون مجوز، عالی برای instruction

# گزینه 2: Qwen (عالی برای فارسی/انگلیسی)
# BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"

# گزینه 3: DialoGPT (برای dialogue - نیاز به فرمت خاص)
# BASE_MODEL = "microsoft/DialoGPT-medium"

# گزینه 4: GPT2 (برای text generation - نیاز به فرمت خاص)
# BASE_MODEL = "gpt2-medium"

# بررسی دسترسی به Hugging Face (اختیاری برای مدل‌های باز)
# Check Hugging Face access (optional for open models)
from huggingface_hub import HfFolder

hf_token = (
    os.getenv("HF_TOKEN") or 
    os.getenv("HUGGINGFACE_TOKEN") or
    HfFolder.get_token()  # خواندن از کش Hugging Face
)

# برای مدل‌های باز، توکن اختیاری است
# For open models, token is optional
if hf_token:
    print("✅ توکن Hugging Face یافت شد (اختیاری)")
    print("✅ Hugging Face token found (optional)")
else:
    print("ℹ️  بدون توکن ادامه می‌دهیم (مدل باز است)")
    print("ℹ️  Continuing without token (model is open)")

# بارگذاری دیتاست
# Load dataset
print("📚 بارگذاری دیتاست...")
print("📚 Loading dataset...")
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"✅ {len(dataset)} نمونه بارگذاری شد")
print(f"✅ {len(dataset)} examples loaded")

# تبدیل به فرمت مناسب (سازگار با Phi-3 و Qwen)
# Convert to proper format (compatible with Phi-3 and Qwen)
def format_prompt(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # فرمت استاندارد instruction following (کار می‌کند با Phi-3, Qwen, و بیشتر مدل‌ها)
    # Standard instruction following format (works with Phi-3, Qwen, and most models)
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
try:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=hf_token,
        trust_remote_code=True
    )
except Exception as e:
    if "gated" in str(e).lower() or "403" in str(e) or "access" in str(e).lower():
        print("❌ خطا: دسترسی به مدل محدود است!")
        print("❌ Error: Model access is restricted!")
        print("")
        print("📝 لطفا این مراحل را انجام دهید:")
        print("📝 Please follow these steps:")
        print("")
        print("1. به این آدرس بروید و مجوز را بپذیرید:")
        print("   Visit and accept the license:")
        print(f"   https://huggingface.co/{BASE_MODEL}")
        print("")
        print("2. مطمئن شوید که با حساب درست لاگین کرده‌اید:")
        print("   Make sure you're logged in with the correct account:")
        print("   huggingface-cli login")
        print("")
        print("3. چند دقیقه صبر کنید تا دسترسی فعال شود")
        print("   Wait a few minutes for access to be activated")
        print("")
        sys.exit(1)
    else:
        raise

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
try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
except Exception as e:
    if "gated" in str(e).lower() or "403" in str(e) or "access" in str(e).lower():
        print("❌ خطا: دسترسی به مدل محدود است!")
        print("❌ Error: Model access is restricted!")
        print("")
        print("📝 لطفا این مراحل را انجام دهید:")
        print("📝 Please follow these steps:")
        print("")
        print("1. به این آدرس بروید و مجوز را بپذیرید:")
        print("   Visit and accept the license:")
        print(f"   https://huggingface.co/{BASE_MODEL}")
        print("")
        print("2. مطمئن شوید که با حساب درست لاگین کرده‌اید:")
        print("   Make sure you're logged in with the correct account:")
        print("   huggingface-cli login")
        print("")
        print("3. چند دقیقه صبر کنید تا دسترسی فعال شود")
        print("   Wait a few minutes for access to be activated")
        print("")
        sys.exit(1)
    else:
        raise

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

# تابع توکنایز کردن با labels
# Tokenization function with labels
def tokenize_function(examples):
    # توکنایز کردن کل متن (prompt + response)
    # Tokenize full text (prompt + response)
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None,
    )
    
    # پیدا کردن طول prompt برای هر نمونه
    # Find prompt length for each example
    texts = examples["text"]
    if isinstance(texts, str):
        texts = [texts]
    
    labels_list = []
    input_ids = tokenized["input_ids"]
    if not isinstance(input_ids[0], list):
        input_ids = [input_ids]
    
    for i, text in enumerate(texts):
        # پیدا کردن موقعیت "### Response:" در متن
        # Find position of "### Response:" in text
        response_marker = "### Response:\n"
        response_pos = text.find(response_marker)
        
        if response_pos != -1:
            # توکنایز کردن فقط prompt (قبل از Response)
            # Tokenize only prompt (before Response)
            prompt_text = text[:response_pos + len(response_marker)]
            prompt_tokenized = tokenizer(
                prompt_text,
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None,
            )
            prompt_length = len(prompt_tokenized["input_ids"])
        else:
            # اگر Response پیدا نشد، نصف متن را prompt در نظر بگیر
            # If Response not found, consider half text as prompt
            prompt_length = len(input_ids[i]) // 2
        
        # ایجاد labels: فقط بخش response باید loss داشته باشد
        # Create labels: only response part should have loss
        labels = list(input_ids[i].copy())
        
        # قسمت prompt را در labels به -100 تبدیل می‌کنیم (ignore index)
        # Convert prompt part in labels to -100 (ignore index)
        for j in range(min(prompt_length, len(labels))):
            labels[j] = -100
        
        labels_list.append(labels)
    
    tokenized["labels"] = labels_list
    
    # حذف فیلد text (دیگر نیاز نیست)
    # Remove text field (no longer needed)
    if "text" in tokenized:
        del tokenized["text"]
    
    return tokenized

# توکنایز کردن دیتاست
# Tokenize dataset
print("🔤 توکنایز کردن دیتاست...")
print("🔤 Tokenizing dataset...")
train_dataset = train_dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=["text"]  # حذف فیلد text بعد از tokenization
)
eval_dataset = eval_dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=["text"]  # حذف فیلد text بعد از tokenization
)

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
    remove_unused_columns=False,  # برای حفظ labels
    dataloader_pin_memory=False,  # برای جلوگیری از مشکلات حافظه
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

