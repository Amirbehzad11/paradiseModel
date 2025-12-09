#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
چت بات ترمینالی - حلقه بی‌نهایت
Terminal chat bot - infinite loop
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from huggingface_hub import HfFolder
import sys

# بررسی وجود مدل
# Check if model exists
if not os.path.exists("./final_model"):
    print("❌ خطا: مدل آموزش‌دیده یافت نشد!")
    print("❌ Error: Trained model not found!")
    print("📝 لطفا ابتدا train_once.py را اجرا کنید")
    print("📝 Please run train_once.py first")
    sys.exit(1)

print("🤖 بارگذاری مدل...")
print("🤖 Loading model...")

# تنظیمات quantization
# Quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# بارگذاری مدل پایه - باید با train_once.py یکسان باشد
# Load base model - must match train_once.py
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # کاملاً باز، بدون مجوز
# BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"  # جایگزین: پشتیبانی عالی از فارسی

# برای مدل‌های باز، توکن اختیاری است
# For open models, token is optional
from huggingface_hub import HfFolder
hf_token = (
    os.getenv("HF_TOKEN") or 
    os.getenv("HUGGINGFACE_TOKEN") or
    HfFolder.get_token()
)

print("📥 بارگذاری مدل پایه...")
print("📥 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# بارگذاری LoRA weights
# Load LoRA weights
print("📥 بارگذاری وزن‌های LoRA...")
print("📥 Loading LoRA weights...")
model = PeftModel.from_pretrained(base_model, "./final_model")
# برای 4-bit quantization، merge ممکن است مشکلاتی ایجاد کند
# For 4-bit quantization, merge may cause issues
# استفاده مستقیم از LoRA بدون merge (بهتر برای 4-bit)
# Use LoRA directly without merge (better for 4-bit)
print("✅ LoRA weights loaded (using without merge for better compatibility)")

# بارگذاری توکنایزر
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./final_model", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("✅ مدل آماده است!")
print("✅ Model ready!")
print("=" * 50)
print("💬 چت بات آماده است. برای خروج 'خروج' یا 'exit' تایپ کنید")
print("💬 Chat bot ready. Type 'خروج' or 'exit' to quit")
print("=" * 50)

# حلقه چت بی‌نهایت
# Infinite chat loop
while True:
    try:
        # دریافت ورودی کاربر
        # Get user input
        user_input = input("\n👤 شما: ").strip()
        
        # بررسی خروج
        # Check exit
        if user_input.lower() in ["خروج", "exit", "quit", "q"]:
            print("\n👋 خداحافظ!")
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # فرمت کردن prompt (فرمت استاندارد instruction - همان فرمت آموزش)
        # Format prompt (standard instruction format - same as training)
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        
        # توکنایز کردن
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # تولید پاسخ
        # Generate response
        print("🤖 مدل: ", end="", flush=True)
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,  # کاهش برای پاسخ‌های کوتاه‌تر
                    temperature=0.3,  # کاهش برای پاسخ‌های دقیق‌تر
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # افزایش برای جلوگیری از تکرار
                    no_repeat_ngram_size=3,  # جلوگیری از تکرار n-gram
                )
            except Exception as e:
                print(f"\n❌ خطا در تولید: {e}")
                print("❌ Error in generation: {e}")
                continue
        
        # دیکد کردن پاسخ
        # Decode response
        if len(outputs) == 0 or len(outputs[0]) == 0:
            print("(پاسخ خالی)")
            print("(Empty response)")
            continue
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # استخراج فقط بخش پاسخ
        # Extract only response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        else:
            # اگر فرمت نبود، فقط بخش جدید را بگیر (بعد از prompt)
            # If format not found, take only new part (after prompt)
            input_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        if not response:
            print("(پاسخ خالی)")
            print("(Empty response)")
        else:
            print(response)
        
    except KeyboardInterrupt:
        print("\n\n👋 خداحافظ!")
        print("👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ خطا: {e}")
        print(f"❌ Error: {e}")
        continue

