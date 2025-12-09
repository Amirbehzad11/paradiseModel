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

# بارگذاری مدل پایه
# Load base model
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # باید با train_once.py یکسان باشد
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

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
model = model.merge_and_unload()  # ادغام LoRA با مدل پایه برای سرعت بیشتر

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
        
        # فرمت کردن prompt
        # Format prompt
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # دیکد کردن پاسخ
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # استخراج فقط بخش پاسخ
        # Extract only response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        else:
            # اگر فرمت نبود، کل پاسخ را بگیر
            # If format not found, take full response
            response = response[len(prompt):].strip()
        
        print(response)
        
    except KeyboardInterrupt:
        print("\n\n👋 خداحافظ!")
        print("👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ خطا: {e}")
        print(f"❌ Error: {e}")
        continue

