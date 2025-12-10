#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سرویس چت - منطق اصلی تولید پاسخ
Chat Service - Main response generation logic
"""
import torch
from app.core.model_loader import get_model
from app.core.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_NO_REPEAT_NGRAM_SIZE
)


class ChatService:
    """سرویس چت برای تولید پاسخ"""
    
    def __init__(self):
        self.peft_model = None
        self.tokenizer = None
    
    def _ensure_model_loaded(self):
        """اطمینان از بارگذاری مدل"""
        if self.peft_model is None or self.tokenizer is None:
            self.peft_model, self.tokenizer = get_model()
    
    async def generate_response(
        self,
        message: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    ) -> str:
        """
        تولید پاسخ برای پیام کاربر
        
        Args:
            message: پیام کاربر
            max_tokens: حداکثر تعداد token
            temperature: کنترل خلاقیت
            top_p: Nucleus sampling
            top_k: تعداد کلمات برتر
            repetition_penalty: جلوگیری از تکرار
            no_repeat_ngram_size: جلوگیری از تکرار n-gram
        
        Returns:
            پاسخ مدل
        """
        self._ensure_model_loaded()
        
        # ساخت prompt بهتر با context بیشتر
        # استفاده از فرمت مشابه دیتاست برای سازگاری بیشتر
        prompt = f"User: {message}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.peft_model.device)
        
        # Generate با پارامترهای بهبود یافته
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_length=20,  # حداقل طول برای پاسخ‌های کامل
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True,  # توقف زودتر برای پاسخ‌های بهتر
            )
        
        # Decode response
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()
        
        # حذف prompt اگر در response باشد
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response

