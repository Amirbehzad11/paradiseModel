#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سیستم آموزش تدریجی - هر بار دیتاست بزرگتر می‌شود
Incremental training system - dataset grows each time
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

BASE_MODEL = "HooshvareLab/gpt2-fa"
DATASET_FILE = "dataset.json"
BACKUP_DATASET = "dataset_backup.json"
FINAL_MODEL_DIR = "./final_model"
NEW_CHATS_FILE = "new_chats.json"

print("Starting incremental training...")

# پشتیبان‌گیری از دیتاست فعلی
if os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        current_dataset = json.load(f)
    print(f"Current dataset: {len(current_dataset)} examples")
    
    with open(BACKUP_DATASET, "w", encoding="utf-8") as f:
        json.dump(current_dataset, f, ensure_ascii=False, indent=4)
    print(f"Backup saved to {BACKUP_DATASET}")
else:
    print("❌ Dataset not found!")
    sys.exit(1)

# بارگذاری چت‌های جدید از chat_with_learning
if os.path.exists(NEW_CHATS_FILE):
    with open(NEW_CHATS_FILE, "r", encoding="utf-8") as f:
        new_chats = json.load(f)
    
    if new_chats:
        print(f"Found {len(new_chats)} new chats from conversations")
        # تبدیل به فرمت dataset
        for chat in new_chats:
            if "instruction" in chat and "response" in chat:
                current_dataset.append({
                    "instruction": chat["instruction"],
                    "response": chat["response"]
                })
        print(f"Added {len(new_chats)} new examples from chats")
        print(f"Total dataset now: {len(current_dataset)} examples")
        
        # ذخیره dataset به‌روز شده
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            json.dump(current_dataset, f, ensure_ascii=False, indent=2)
        
        # پاک کردن چت‌های استفاده شده
        with open(NEW_CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    else:
        print("No new chats found")
else:
    print("No new chats file found")

# اگر چت‌های جدید اضافه شدند، نیازی به تولید نمونه‌های جدید نیست
# چت‌های واقعی کاربر بهتر از نمونه‌های تولید شده هستند

print(f"Final dataset size: {len(current_dataset)} examples")

# تبدیل به فرمت مناسب
def format_prompt(example):
    instruction = example.get("instruction", "")
    response = example.get("response", "")
    prompt = f"User: {instruction}\nAssistant: {response}"
    return {"text": prompt}

formatted_data = [format_prompt(ex) for ex in current_dataset]
dataset = Dataset.from_list(formatted_data)

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Setting up quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

print("Loading base model...")
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
        return_tensors=None,
    )
    
    labels = []
    for i, text in enumerate(examples["text"]):
        assistant_prefix = "Assistant:"
        assistant_start_idx = text.find(assistant_prefix)
        
        if assistant_start_idx != -1:
            prompt_tokens = tokenizer(
                text[:assistant_start_idx + len(assistant_prefix)],
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            prompt_length = len(prompt_tokens)
        else:
            prompt_length = len(tokenized["input_ids"][i]) // 2
        
        current_labels = list(tokenized["input_ids"][i])
        for j in range(min(prompt_length, len(current_labels))):
            current_labels[j] = -100
        labels.append(current_labels)
    
    tokenized["labels"] = labels
    return tokenized

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=2,  # کاهش به 2 epoch برای training سریع‌تر
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

print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

print("Saving final model...")
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("Training completed successfully!")
print(f"Model saved to {FINAL_MODEL_DIR}")
print(f"Dataset now has {len(current_dataset)} examples")

