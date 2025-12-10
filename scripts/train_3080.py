#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
فاین‌تیون بهینه برای RTX 3080 10GB
Optimized Fine-tuning for RTX 3080 10GB with QLoRA
"""
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import sys

# اضافه کردن مسیر روت
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import DATA_DIR, MODEL_DIR

# تنظیمات
# استفاده از مدل باز بدون نیاز به احراز هویت Hugging Face
# Using open model without Hugging Face authentication requirement
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # کاملاً باز، بدون نیاز به مجوز
DATASET_PATH = DATA_DIR / "data_persian.json"
OUTPUT_DIR = MODEL_DIR / "phi3_mini_finetuned"
CHECKPOINT_DIR = BASE_DIR / "checkpoints_3080"

# تنظیمات QLoRA برای RTX 3080
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
BATCH_SIZE = 3
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 2048

print("=" * 80)
print("Fine-tuning for RTX 3080 10GB")
print("=" * 80)
print(f"Base Model: {BASE_MODEL}")
print(f"✅ Using open model - No authentication required!")
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

# بررسی دیتاست
if not DATASET_PATH.exists():
    print(f"❌ Dataset not found at {DATASET_PATH}")
    print("💡 Run dataset_prep.py first to prepare dataset")
    sys.exit(1)

# بارگذاری دیتاست
print("\n📚 Loading dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"✅ Loaded {len(raw_data)} examples")

# تبدیل به فرمت مناسب برای Phi-3
def format_phi3(examples):
    """تبدیل به فرمت Phi-3"""
    formatted = []
    for item in examples:
        if "messages" in item:
            # فرمت ChatML برای Phi-3
            messages = item["messages"]
            text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    text += f"<|user|>\n{content}<|end|>\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}<|end|>\n"
            formatted.append({"text": text})
        elif "instruction" in item:
            # فرمت Alpaca برای Phi-3
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>\n"
            formatted.append({"text": text})
    return formatted

print("\n🔄 Formatting dataset...")
formatted_data = format_phi3(raw_data)
dataset = Dataset.from_list(formatted_data)

# تقسیم train/eval
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"✅ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# بارگذاری tokenizer
print("\n🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# تنظیم special tokens برای ChatML
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# اضافه کردن special tokens برای ChatML
special_tokens = {
    "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end|>"]
}
tokenizer.add_special_tokens(special_tokens)

# Tokenization
print("\n🔤 Tokenizing dataset...")
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    
    # برای causal LM، labels همان input_ids است
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing train"
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing eval"
)

# تنظیمات Quantization برای RTX 3080
print("\n⚙️ Setting up 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# بارگذاری مدل
print("\n🤖 Loading base model (this may take a few minutes)...")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

# بررسی نصب flash_attention
try:
    import flash_attn
    use_flash_attention = torch.cuda.is_available()
    if use_flash_attention:
        print("✅ Flash Attention 2 detected")
except ImportError:
    use_flash_attention = False
    print("ℹ️  Flash Attention 2 not installed, using eager attention")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,  # استفاده از dtype به جای torch_dtype (deprecated)
    attn_implementation="flash_attention_2" if use_flash_attention else "eager",
)

# آماده‌سازی برای training
model = prepare_model_for_kbit_training(model)

# تنظیم LoRA
print("\n🔧 Setting up LoRA...")
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    modules_to_save=["embed_tokens", "lm_head"],  # برای special tokens
)

model = get_peft_model(model, lora_config)

# Resize token embeddings برای special tokens
model.resize_token_embeddings(len(tokenizer))

# نمایش پارامترهای trainable
model.print_trainable_parameters()

# تنظیمات Training
print("\n⚙️ Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,  # استفاده از bfloat16 برای RTX 3080
    fp16=False,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,  # صرفه‌جویی در حافظه
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
print("\n🎓 Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# آموزش
print("\n" + "=" * 80)
print("🚀 Starting training...")
print("=" * 80)
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"Total steps: {len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION) * NUM_EPOCHS}")
print("=" * 80)

trainer.train()

# ذخیره مدل
print("\n💾 Saving final model...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ذخیره adapter
model.save_pretrained(str(OUTPUT_DIR))

# ذخیره tokenizer
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"\n✅ Model saved to {OUTPUT_DIR}")
print("=" * 80)
print("Training completed successfully!")
print("=" * 80)

