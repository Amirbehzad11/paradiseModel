#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سیستم یادگیری پیوسته - مدل همیشه در حال یادگیری
Continuous Learning System - Model Always Learning
"""
import os
import json
import torch
import random
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
import sys

BASE_MODEL = "HooshvareLab/gpt2-fa"
DATASET_FILE = "dataset.json"
BACKUP_DATASET = "dataset_backup.json"
FINAL_MODEL_DIR = "./final_model"

# تنظیمات
GENERATE_NEW_EXAMPLES = 100  # تعداد نمونه‌های جدید در هر چرخه
MIN_CYCLE_INTERVAL = 300  # حداقل فاصله بین چرخه‌ها (ثانیه)
MAX_DATASET_SIZE = 10000  # حداکثر اندازه dataset

print("=" * 60)
print("🚀 Continuous Learning System Started")
print("=" * 60)

def load_dataset():
    """بارگذاری dataset"""
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"📚 Dataset loaded: {len(dataset)} examples")
        return dataset
    else:
        print("❌ Dataset not found!")
        sys.exit(1)

def backup_dataset(dataset):
    """پشتیبان‌گیری از dataset"""
    with open(BACKUP_DATASET, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"💾 Backup saved: {len(dataset)} examples")

def generate_new_examples(model, tokenizer, num_examples=GENERATE_NEW_EXAMPLES):
    """تولید نمونه‌های جدید از مدل"""
    print(f"\n🎨 Generating {num_examples} new examples...")
    
    new_examples = []
    
    # الگوهای متنوع برای تولید
    patterns = [
        ("این عکس {rel}مه که سال {year} فوت کرد. می‌خوام باهاش حرف بزنم", "پدر", "مادر", "برادر", "خواهر", "همسر", "فرزند"),
        ("{rel}م سال {year} فوت کرد. می‌خوام باهاش صحبت کنم", "پدر", "مادر", "برادر", "خواهر", "همسر", "فرزند"),
        ("سلام {rel}م", "پدر", "مادر", "برادر", "خواهر", "همسر", "فرزند"),
        ("سلام {rel}م، خوبی؟", "پدر", "مادر", "برادر", "خواهر", "همسر", "فرزند"),
        ("تو {rel} منی", "پدر", "مادر", "برادر", "خواهر", "همسر", "فرزند"),
        ("می‌خوام با {rel}م صحبت کنم", "پدر", "مادر", "برادر", "خواهر", "همسر", "فرزند"),
        ("یادت چقدر {action} می‌کردیم؟", "شیطنت", "بازی", "خوش می‌گذروندیم", "حرف می‌زدیم", "خندیدیم"),
        ("زندگی خوبی داری؟", None),
        ("تو چطوری زندگی داری؟", None),
        ("می‌تونی در مورد خودت بگی؟", None),
        ("می‌تونی در مورد زندگی بهم بگی؟", None),
        ("درباره مرگ و زندگی پس از آن توضیح بده", None),
        ("بهشت چگونه جایی است؟", None),
        ("چگونه می‌توانم آرامش معنوی پیدا کنم؟", None),
    ]
    
    years = ["۹۵", "۹۶", "۹۷", "۹۸", "۹۹", "۱۴۰۰", "۱۴۰۱", "۱۴۰۲", "۱۴۰۳"]
    actions = ["شیطنت", "بازی", "خوش می‌گذروندیم", "حرف می‌زدیم", "خندیدیم", "سفر می‌کردیم"]
    
    device = next(model.parameters()).device
    
    generated = 0
    attempts = 0
    max_attempts = num_examples * 3  # حداکثر تلاش
    
    while generated < num_examples and attempts < max_attempts:
        attempts += 1
        
        # انتخاب الگو
        pattern_template, *rel_options = random.choice(patterns)
        
        # ساخت instruction
        if "{rel}" in pattern_template:
            if rel_options and rel_options[0]:
                rel = random.choice(rel_options)
                if "{year}" in pattern_template:
                    year = random.choice(years)
                    instruction = pattern_template.format(rel=rel, year=year)
                else:
                    instruction = pattern_template.format(rel=rel)
            else:
                continue
        elif "{action}" in pattern_template:
            action = random.choice(actions)
            instruction = pattern_template.format(action=action)
        else:
            instruction = pattern_template
        
        # تولید پاسخ
        prompt = f"User: {instruction}\nAssistant:"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
            
            input_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            # فیلتر کردن پاسخ‌های خوب
            if (response and 
                len(response) > 30 and 
                len(response) < 600 and
                not response.startswith("User:") and
                not response.startswith("Assistant:") and
                any(ord(c) > 127 for c in response)):  # حتماً فارسی باشد
                
                new_examples.append({
                    "instruction": instruction,
                    "response": response
                })
                generated += 1
                
                if generated % 10 == 0:
                    print(f"  ✓ Generated {generated}/{num_examples} examples...")
        
        except Exception as e:
            continue
    
    print(f"✅ Generated {len(new_examples)} valid examples")
    return new_examples

def train_model(dataset):
    """آموزش مدل"""
    print(f"\n🎓 Starting training with {len(dataset)} examples...")
    
    # تبدیل به فرمت مناسب
    def format_prompt(example):
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        prompt = f"User: {instruction}\nAssistant: {response}"
        return {"text": prompt}
    
    formatted_data = [format_prompt(ex) for ex in dataset]
    train_dataset = Dataset.from_list(formatted_data)
    
    # تقسیم train/eval
    train_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]
    
    print(f"📊 Split: {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    
    # Load model
    if os.path.exists(FINAL_MODEL_DIR) and os.path.exists(f"{FINAL_MODEL_DIR}/adapter_config.json"):
        print("📥 Loading existing model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        model = PeftModel.from_pretrained(base_model, FINAL_MODEL_DIR)
        model = prepare_model_for_kbit_training(model)
        
        # بررسی و فعال کردن trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            print("⚠️ Warning: No trainable parameters! Enabling training for LoRA layers...")
            # فعال کردن training برای همه LoRA parameters
            for name, param in model.named_parameters():
                if 'lora' in name.lower() or 'adapter' in name.lower():
                    param.requires_grad = True
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ Enabled {trainable_params:,} trainable parameters")
        
        model.train()
    else:
        print("📥 Loading base model...")
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
    
    # بررسی نهایی trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Final check - Trainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.4f}")
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters found! Cannot train model.")
    
    # Tokenize
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
    
    print("🔤 Tokenizing...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Training args
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        learning_rate=2e-4,
        fp16=False,  # غیرفعال کردن fp16 برای جلوگیری از مشکل optimizer
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    print("🚀 Training started...")
    trainer.train()
    
    # Save
    print("💾 Saving model...")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    print("✅ Training completed!")
    return model, tokenizer

def main_loop():
    """حلقه اصلی یادگیری پیوسته"""
    cycle = 0
    
    while True:
        cycle += 1
        print("\n" + "=" * 60)
        print(f"🔄 CYCLE {cycle} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 1. بارگذاری dataset
        dataset = load_dataset()
        
        # 2. پشتیبان‌گیری
        backup_dataset(dataset)
        
        # 3. بارگذاری مدل برای تولید نمونه‌های جدید
        if os.path.exists(FINAL_MODEL_DIR) and os.path.exists(f"{FINAL_MODEL_DIR}/adapter_config.json"):
            print("\n📥 Loading model for generation...")
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
            model = PeftModel.from_pretrained(base_model, FINAL_MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR, trust_remote_code=True)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 4. تولید نمونه‌های جدید
            new_examples = generate_new_examples(model, tokenizer, GENERATE_NEW_EXAMPLES)
            
            # 5. اضافه کردن به dataset
            if new_examples:
                dataset.extend(new_examples)
                
                # محدود کردن اندازه dataset
                if len(dataset) > MAX_DATASET_SIZE:
                    print(f"📉 Dataset too large ({len(dataset)}), keeping last {MAX_DATASET_SIZE} examples")
                    dataset = dataset[-MAX_DATASET_SIZE:]
                
                # ذخیره dataset
                with open(DATASET_FILE, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"📚 Dataset updated: {len(dataset)} examples")
            
            # پاک کردن مدل از حافظه
            del model
            del base_model
            torch.cuda.empty_cache()
        
        # 6. آموزش مدل
        model, tokenizer = train_model(dataset)
        
        # پاک کردن از حافظه
        del model
        torch.cuda.empty_cache()
        
        # 7. انتظار قبل از چرخه بعدی
        print(f"\n⏳ Waiting {MIN_CYCLE_INTERVAL} seconds before next cycle...")
        print("   (Press Ctrl+C to stop)")
        time.sleep(MIN_CYCLE_INTERVAL)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\n👋 Continuous learning stopped by user")
        print("✅ All progress saved!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

