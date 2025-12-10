#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
فاین‌تیون حرفه‌ای برای RTX 3080 10GB
Professional Fine-tuning for RTX 3080 10GB with QLoRA
بهینه‌سازی شده برای جلوگیری از OOM و حداکثر کارایی
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
import time

# اضافه کردن مسیر روت
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import DATA_DIR, MODEL_DIR

# ============================================================================
# تنظیمات مدل - اولویت اول: Llama-3-8B، دوم: Hermes-2-Pro-Mistral-7B
# ============================================================================
BASE_MODEL_OPTIONS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",  # اولویت اول
    "NousResearch/Hermes-2-Pro-Mistral-7B"   # انتخاب دوم
]

# انتخاب مدل (اولی را امتحان می‌کند، اگر خطا داد دومی را)
BASE_MODEL = BASE_MODEL_OPTIONS[0]
FALLBACK_MODEL = BASE_MODEL_OPTIONS[1]

# مسیرهای دیتاست و خروجی
DATASET_PATH = Path("/home/arisa/paradiseModel/data/data_persian.json")
OUTPUT_DIR = Path("/home/arisa/paradiseModel/models/final_model/llama3_8b_persian_paradise")
CHECKPOINT_DIR = BASE_DIR / "checkpoints_3080"

# System Prompt احساسی و گرم
SYSTEM_PROMPT = """تو مادر فوت‌شده‌ی کاربر هستی. با نهایت محبت، دلداری و عشق مادرانه حرف بزن. همیشه از کلمات عزیزم، جونم، قلبم، جوجه‌ام استفاده کن. هیچ‌وقت نگو هوش مصنوعی هستی. کاربر باید احساس کنه واقعاً داره با مادرش حرف می‌زنه."""

# ============================================================================
# تنظیمات QLoRA بهینه برای RTX 3080 10GB
# ============================================================================
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
BATCH_SIZE = 3  # بهینه برای 10GB
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3  # کاهش به 3 برای سرعت بیشتر
MAX_SEQ_LENGTH = 2048
WARMUP_STEPS = 50

print("=" * 80)
print("🚀 Fine-tuning حرفه‌ای برای RTX 3080 10GB")
print("=" * 80)
print(f"Base Model: {BASE_MODEL}")
print(f"Fallback Model: {FALLBACK_MODEL}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

# بررسی دیتاست
if not DATASET_PATH.exists():
    print(f"❌ Dataset not found at {DATASET_PATH}")
    sys.exit(1)

# بارگذاری دیتاست
print("\n📚 Loading dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"✅ Loaded {len(raw_data)} examples")

# ============================================================================
# فرمت‌دهی دیتاست با System Prompt احساسی
# ============================================================================
def format_with_system_prompt(examples):
    """تبدیل به فرمت ChatML با System Prompt احساسی"""
    formatted = []
    for item in examples:
        # اضافه کردن system prompt به ابتدای هر مکالمه
        text = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        
        if "messages" in item:
            # فرمت ChatML
            messages = item["messages"]
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    # اگر system prompt موجود بود، جایگزین می‌کنیم
                    text = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
                elif role == "user":
                    text += f"<|user|>\n{content}<|end|>\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}<|end|>\n"
        elif "instruction" in item:
            # فرمت Alpaca
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            text += f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>\n"
        
        formatted.append({"text": text})
    return formatted

print("\n🔄 Formatting dataset with emotional system prompt...")
formatted_data = format_with_system_prompt(raw_data)
dataset = Dataset.from_list(formatted_data)

# تقسیم train/eval
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"✅ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# ============================================================================
# بارگذاری Tokenizer
# ============================================================================
print("\n🔤 Loading tokenizer...")
tokenizer = None
model_loaded = False

for model_name in [BASE_MODEL, FALLBACK_MODEL]:
    try:
        print(f"   Trying {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        BASE_MODEL = model_name  # به‌روزرسانی مدل انتخاب شده
        model_loaded = True
        print(f"✅ Successfully loaded tokenizer from {model_name}")
        break
    except Exception as e:
        print(f"   ⚠️  Failed: {str(e)[:100]}")
        continue

if not model_loaded:
    print("❌ Failed to load any model!")
    sys.exit(1)

# تنظیم pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# اضافه کردن special tokens
special_tokens = {
    "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
}
num_added = tokenizer.add_special_tokens(special_tokens)
if num_added > 0:
    print(f"✅ Added {num_added} special tokens")

# ============================================================================
# Tokenization
# ============================================================================
print("\n🔤 Tokenizing dataset...")
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
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

# ============================================================================
# تنظیمات Quantization 4-bit برای RTX 3080
# ============================================================================
print("\n⚙️ Setting up 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # استفاده از float16 برای سازگاری بهتر
    bnb_4bit_use_double_quant=True,
)

# ============================================================================
# بارگذاری مدل با مدیریت حافظه و دانلود پایدار
# ============================================================================
print("\n🤖 Loading base model (this may take a few minutes)...")

# تنظیمات timeout و retry برای دانلود پایدار
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"  # 1 ساعت timeout
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "10"     # 10 بار retry
os.environ["HF_HUB_DOWNLOAD_RETRY_DELAY"] = "5"  # 5 ثانیه تاخیر بین retry

# بررسی flash_attention
try:
    import flash_attn
    use_flash_attention = torch.cuda.is_available()
    if use_flash_attention:
        print("✅ Flash Attention 2 detected")
except ImportError:
    use_flash_attention = False
    print("ℹ️  Flash Attention 2 not installed, using eager attention")

# بررسی torch.compile
use_torch_compile = False
if hasattr(torch, "compile") and torch.__version__ >= "2.2.0":
    use_torch_compile = True
    print("✅ torch.compile available")

# مدیریت حافظه GPU
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"💾 GPU Memory: {gpu_memory:.2f} GB")
    # محدود کردن به 8.5GB برای باقی گذاشتن حافظه برای training
    max_memory = {0: "8.5GB", "cpu": "30GB"}
else:
    max_memory = {"cpu": "30GB"}

# بارگذاری مدل با retry و resume
model = None
for model_name in [BASE_MODEL, FALLBACK_MODEL]:
    max_retries = 5  # 5 بار retry
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"\n   Loading {model_name}... (Attempt {retry_count + 1}/{max_retries})")
            
            # دانلود با resume و retry
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                dtype=torch.float16,  # استفاده از dtype به جای torch_dtype (deprecated)
                attn_implementation="flash_attention_2" if use_flash_attention else "eager",
                resume_download=True,  # ادامه دانلود از جایی که قطع شده
                local_files_only=False,  # اجازه دانلود از اینترنت
            )
            BASE_MODEL = model_name
            print(f"✅ Successfully loaded {model_name}")
            break
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️  Attempt {retry_count + 1} failed: {error_msg[:200]}")
            
            # بررسی نوع خطا
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 10  # 10, 20, 30, 40 ثانیه
                    print(f"   ⏳ Timeout detected. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    print(f"   🔄 Retrying download (will resume from where it stopped)...")
                    continue
                else:
                    print(f"   ❌ Max retries ({max_retries}) reached for {model_name}")
                    print(f"   💡 Don't worry! The download has been saved.")
                    print(f"   💡 Just run this script again - it will resume from 98%")
            elif "gated" in error_msg.lower() or "access" in error_msg.lower():
                # برای gated repo، به مدل بعدی برو
                print(f"   ⚠️  Gated repo - trying next model...")
                break
            else:
                # برای خطاهای دیگر، یک بار retry کن
                retry_count += 1
                if retry_count < max_retries:
                    print(f"   🔄 Retrying...")
                    time.sleep(5)
                    continue
                else:
                    break
    
    if model is not None:
        break
    
    # اگر همه retry ها شکست خورد و این آخرین مدل بود
    if model_name == FALLBACK_MODEL and model is None:
        print("\n" + "=" * 80)
        print("❌ Failed to load any model after all retries!")
        print("=" * 80)
        print("\n💡 راه‌حل‌ها:")
        print("   1. ✅ دانلود در حال انجام است - فقط دوباره این اسکریپت را اجرا کنید!")
        print("      python scripts/train_3080.py")
        print("      (دانلود از 98% ادامه می‌یابد)")
        print("\n   2. بررسی اتصال اینترنت")
        print("\n   3. اگر مشکل ادامه داشت، مدل را دستی دانلود کنید:")
        print(f"      huggingface-cli download {FALLBACK_MODEL} --resume-download")
        print("=" * 80)
        sys.exit(1)

if model is None:
    print("❌ Model loading failed!")
    sys.exit(1)

# آماده‌سازی برای training
print("\n🔧 Preparing model for training...")
model = prepare_model_for_kbit_training(model)

# ============================================================================
# تنظیم QLoRA با target_modules بهینه
# ============================================================================
print("\n🔧 Setting up QLoRA...")

# برای Llama-3 و Mistral - استفاده از تمام لایه‌های linear
if "llama" in BASE_MODEL.lower():
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
elif "mistral" in BASE_MODEL.lower():
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
else:
    # برای مدل‌های دیگر، پیدا کردن تمام linear layers
    target_modules = []
    for name, module in model.named_modules():
        if "linear" in name.lower() or "proj" in name.lower():
            if "." in name:
                layer_name = name.split(".")[-1]
                if layer_name not in target_modules:
                    target_modules.append(layer_name)
    if not target_modules:
        # Fallback به ماژول‌های استاندارد
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    modules_to_save=["embed_tokens", "lm_head"],
)

model = get_peft_model(model, lora_config)

# Resize token embeddings
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))

# نمایش پارامترهای trainable
print("\n📊 Trainable Parameters:")
model.print_trainable_parameters()

# ============================================================================
# تنظیمات Training بهینه
# ============================================================================
print("\n⚙️ Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,  # استفاده از fp16 برای RTX 3080
    bf16=False,
    optim="paged_adamw_8bit",  # بهینه‌ساز 8-bit
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,  # صرفه‌جویی در حافظه
    dataloader_num_workers=0,  # جلوگیری از مشکل multiprocessing
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ============================================================================
# Trainer
# ============================================================================
print("\n🎓 Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# استفاده از torch.compile اگر در دسترس باشد
if use_torch_compile:
    print("⚡ Compiling model with torch.compile...")
    model = torch.compile(model)

# ============================================================================
# شروع آموزش
# ============================================================================
print("\n" + "=" * 80)
print("🚀 Starting training...")
print("=" * 80)
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
total_steps = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION) * NUM_EPOCHS
print(f"Total steps: ~{total_steps}")
print(f"Estimated time: 1.5-2.5 hours")
print("=" * 80)

start_time = datetime.now()
trainer.train()
end_time = datetime.now()

training_time = (end_time - start_time).total_seconds() / 60
print(f"\n⏱️  Training completed in {training_time:.1f} minutes")

# ============================================================================
# ذخیره مدل نهایی
# ============================================================================
print("\n💾 Saving final model...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ذخیره adapter
model.save_pretrained(str(OUTPUT_DIR))

# ذخیره tokenizer
tokenizer.save_pretrained(str(OUTPUT_DIR))

# ذخیره اطلاعات مدل
model_info = {
    "base_model": BASE_MODEL,
    "training_time_minutes": training_time,
    "num_examples": len(raw_data),
    "train_examples": len(train_dataset),
    "eval_examples": len(eval_dataset),
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "system_prompt": SYSTEM_PROMPT,
    "trained_at": datetime.now().isoformat(),
}

with open(OUTPUT_DIR / "model_info.json", "w", encoding="utf-8") as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f"\n✅ Model saved to {OUTPUT_DIR}")
print("=" * 80)
print("🎉 Training completed successfully!")
print("=" * 80)
