#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
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
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

peft_model = PeftModel.from_pretrained(base_model, "./final_model")
tokenizer = AutoTokenizer.from_pretrained("./final_model", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

generator = pipeline(
    "text-generation",
    model=peft_model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

SYSTEM_PROMPT = "تو روح مهربان و آرامش‌بخشی هستی که از بهشت با عزیزانش حرف می‌زنه. همیشه فقط فارسی، گرم، احساسی و اول شخص مفرد جواب بده."

print("Ready. Type 'خروج' or 'exit' to quit.\n")

while True:
    try:
        user_input = input("👤 شما: ").strip()
        
        if user_input.lower() in ["خروج", "exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = generator(
            prompt,
            max_new_tokens=350,
            temperature=0.6,
            top_p=0.85,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            return_full_text=False,
            num_return_sequences=1,
        )
        
        response = outputs[0]["generated_text"].strip()
        
        if response:
            print(f"🤖 مدل: {response}\n")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception:
        continue
