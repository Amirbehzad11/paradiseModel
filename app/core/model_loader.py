#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بارگذاری مدل و tokenizer
Model and Tokenizer Loader
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
from app.core.config import (
    BASE_MODEL,
    MODEL_DIR,
    USE_4BIT,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_COMPUTE_DTYPE
)

# Global variables
peft_model = None
tokenizer = None


def load_model():
    """
    بارگذاری مدل و tokenizer
    
    Returns:
        tuple: (peft_model, tokenizer)
    """
    global peft_model, tokenizer
    
    if peft_model is not None and tokenizer is not None:
        return peft_model, tokenizer
    
    # بررسی وجود مدل
    model_path = MODEL_DIR
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run scripts/train_once.py first."
        )
    
    print("🔄 Loading model...")
    
    # تنظیمات Quantization
    if USE_4BIT:
        compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE, torch.float16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # بارگذاری base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    
    # بارگذاری PEFT model
    peft_model = PeftModel.from_pretrained(base_model, str(model_path))
    
    # بارگذاری tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    
    # تنظیم pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✅ Model loaded successfully!")
    
    return peft_model, tokenizer


def get_model():
    """دریافت مدل (اگر بارگذاری نشده، بارگذاری می‌کند)"""
    global peft_model, tokenizer
    if peft_model is None or tokenizer is None:
        peft_model, tokenizer = load_model()
    return peft_model, tokenizer


def is_model_loaded():
    """بررسی اینکه آیا مدل بارگذاری شده است"""
    return peft_model is not None and tokenizer is not None

