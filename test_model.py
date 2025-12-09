#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اسکریپت تست سریع مدل
Quick model test script
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("🔍 Testing model...")

# تنظیمات
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# بارگذاری
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(base_model, "./final_model")
tokenizer = AutoTokenizer.from_pretrained("./final_model", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# تست با یک نمونه از دیتاست
test_prompt = "### Instruction:\nبهشت چگونه جایی است؟\n\n### Response:\n"

print(f"\n📝 Test prompt: {test_prompt}")

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,
        top_p=0.85,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n🤖 Response:\n{response}")

if "### Response:" in response:
    extracted = response.split("### Response:")[-1].strip()
    print(f"\n✅ Extracted response:\n{extracted}")

