#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تنظیمات اصلی برنامه
Application Configuration
"""
import os
from pathlib import Path

# مسیرهای اصلی
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models" / "final_model"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# ایجاد فولدرها در صورت عدم وجود
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# تنظیمات مدل
BASE_MODEL = os.getenv("BASE_MODEL", "HooshvareLab/gpt2-fa")
MODEL_PATH = MODEL_DIR  # استفاده از Path object

# تنظیمات API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# تنظیمات CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# تنظیمات Generation
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", 100))  # کاهش برای سرعت بیشتر
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 1.0))  # افزایش برای تنوع بیشتر
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", 0.92))  # کاهش جزئی برای کیفیت بهتر
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 40))  # کاهش برای انتخاب بهتر
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", 1.5))  # افزایش برای جلوگیری از تکرار
DEFAULT_NO_REPEAT_NGRAM_SIZE = int(os.getenv("DEFAULT_NO_REPEAT_NGRAM_SIZE", 4))  # افزایش برای جلوگیری از تکرار عبارات

# تنظیمات Device (GPU یا CPU)
USE_CPU = os.getenv("USE_CPU", "false").lower() == "true"  # استفاده از CPU به جای GPU

# تنظیمات Quantization (فقط برای GPU)
USE_4BIT = os.getenv("USE_4BIT", "true").lower() == "true" and not USE_CPU
BNB_4BIT_QUANT_TYPE = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4")
BNB_4BIT_COMPUTE_DTYPE = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "float16")

