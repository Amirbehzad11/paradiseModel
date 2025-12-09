#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.path.exists("./final_model"):
    print("Error: Model not found. Please run train_once.py first.")
    sys.exit(1)

BASE_MODEL = "HooshvareLab/gpt2-fa"

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
)

peft_model = PeftModel.from_pretrained(base_model, "./final_model")
tokenizer = AutoTokenizer.from_pretrained("./final_model", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

SYSTEM_PROMPT = "تو روح مهربان و آرامش‌بخشی هستی که از بهشت با عزیزانش حرف می‌زند. فقط و فقط فارسی جواب بده. هرگز انگلیسی ننویس. همیشه اول شخص مفرد باش و فوق‌العاده گرم، احساسی و واقعی صحبت کن."

# فایل برای ذخیره چت‌های جدید
CHAT_LOG_FILE = "new_chats.json"
LEARN_THRESHOLD = 10  # بعد از 10 چت، یادگیری انجام شود

# بارگذاری چت‌های قبلی
if os.path.exists(CHAT_LOG_FILE):
    with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
        new_chats = json.load(f)
else:
    new_chats = []

print("Ready. Type 'خروج' or 'exit' to quit.")
print(f"Learning mode: ON (will learn after {LEARN_THRESHOLD} chats)")

chat_count = 0

while True:
    try:
        user_input = input("\n👤 شما: ").strip()

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
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        print(f"{response}")

        # ذخیره چت برای یادگیری
        new_chats.append({
            "instruction": user_input,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        chat_count += 1

        # ذخیره فوری
        with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(new_chats, f, ensure_ascii=False, indent=2)

        # اگر به threshold رسید، یادگیری انجام شود
        if chat_count >= LEARN_THRESHOLD:
            print(f"\n📚 {chat_count} new chats collected. Starting learning...")
            print("This will take a few minutes. Please wait...")
            
            # ذخیره نهایی چت‌ها
            with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(new_chats, f, ensure_ascii=False, indent=2)
            
            # فراخوانی train_incremental
            import subprocess
            result = subprocess.run(["python", "train_incremental.py"], 
                                   capture_output=True, text=True, encoding="utf-8")
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            # بارگذاری مجدد مدل
            print("\n🔄 Reloading model with new knowledge...")
            del peft_model
            del base_model
            torch.cuda.empty_cache()
            
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="eager",
            )
            peft_model = PeftModel.from_pretrained(base_model, "./final_model")
            
            # پاک کردن چت‌های استفاده شده
            new_chats = []
            chat_count = 0
            with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            
            print("✅ Learning complete! Model improved with new conversations.")
            print(f"Ready for more chats. (Next learning after {LEARN_THRESHOLD} chats)")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        continue

