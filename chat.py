#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.path.exists("./final_model"):
    print("❌ Model not found. Run train_once.py first.")
    sys.exit(1)

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

BASE_MODEL = "HooshvareLab/gpt2-fa"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

peft_model = PeftModel.from_pretrained(base_model, "./final_model")
tokenizer = AutoTokenizer.from_pretrained("./final_model", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Ready. Type 'خروج' or 'exit' to quit.\n")

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
