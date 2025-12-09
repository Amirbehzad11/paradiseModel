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

# اگر مدل از قبل وجود دارد، پاسخ‌های جدید تولید کن
if os.path.exists(FINAL_MODEL_DIR) and os.path.exists(f"{FINAL_MODEL_DIR}/config.json"):
    print("Model exists. Generating new examples...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=None,
    )
    
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, FINAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # تولید نمونه‌های جدید از مدل
    new_examples = []
    relationships = ["پدر", "مادر", "برادر", "خواهر", "همسر", "فرزند", "پدربزرگ", "مادربزرگ"]
    years = ["۹۵", "۹۶", "۹۷", "۹۸", "۹۹", "۱۴۰۰", "۱۴۰۱", "۱۴۰۲"]
    
    print("Generating new examples from model...")
    for rel in relationships:
        for year in years[:3]:  # فقط 3 سال برای هر رابطه
            inst = f"{rel}م سال {year} فوت کرد. می‌خوام باهاش صحبت کنم"
            prompt = f"User: {inst}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            
            # پیدا کردن device
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            input_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            if response and len(response) > 20 and len(response) < 500:  # فقط پاسخ‌های معنی‌دار
                new_examples.append({
                    "instruction": inst,
                    "response": response
                })
    
    if new_examples:
        print(f"Generated {len(new_examples)} new examples")
        current_dataset.extend(new_examples)
        
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            json.dump(current_dataset, f, ensure_ascii=False, indent=4)
        print(f"Updated dataset: {len(current_dataset)} examples")
    else:
        print("No new examples generated")

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
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
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

