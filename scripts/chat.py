#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اسکریپت چت تعاملی
Interactive chat script
"""
import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings

# اضافه کردن مسیر روت به sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import BASE_MODEL, MODEL_DIR

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = MODEL_DIR

if not MODEL_PATH.exists():
    print(f"❌ Model not found at {MODEL_PATH}. Run scripts/train_once.py first.")
    sys.exit(1)

print("🔄 Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    local_files_only=False,
)

peft_model = PeftModel.from_pretrained(base_model, str(MODEL_PATH))
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("✅ Ready. Type 'خروج' or 'exit' to quit.\n")

while True:
    try:
        user_input = input("👤 شما: ").strip()
        
        if user_input.lower() in ["خروج", "exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        prompt = f"User: {user_input}\nAssistant:"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(peft_model.device)
        
        print("🤖 مدل: ", end="", flush=True)
        
        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=200,  # کاهش برای پاسخ‌های متمرکزتر
                min_length=20,  # حداقل طول برای پاسخ‌های کامل
                temperature=1.0,  # افزایش برای تنوع بیشتر
                top_p=0.92,  # کاهش جزئی برای کیفیت بهتر
                top_k=40,  # کاهش برای انتخاب بهتر
                repetition_penalty=1.5,  # افزایش برای جلوگیری از تکرار
                no_repeat_ngram_size=4,  # افزایش برای جلوگیری از تکرار عبارات
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True,  # توقف زودتر برای پاسخ‌های بهتر
            )
        
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        if response:
            print(f"{response}\n")
        else:
            print("(پاسخ خالی)\n")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ خطا: {e}\n")
        continue

