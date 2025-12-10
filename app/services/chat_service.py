#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سرویس چت - منطق اصلی تولید پاسخ
Chat Service - Main response generation logic
"""
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from app.core.model_loader import get_model
from app.core.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DATA_DIR
)
from app.services.memory_service import MemoryService

CHAT_LOG_PATH = DATA_DIR / "chat_logs.json"


class ChatService:
    """سرویس چت برای تولید پاسخ"""
    
    def __init__(self):
        self.peft_model = None
        self.tokenizer = None
        self.memory = MemoryService()
    
    def _ensure_model_loaded(self):
        """اطمینان از بارگذاری مدل"""
        if self.peft_model is None or self.tokenizer is None:
            self.peft_model, self.tokenizer = get_model()
    
    def _log_chat(self, message: str, response: str):
        """ذخیره چت در لاگ برای یادگیری (در background thread برای سرعت)"""
        # اجرای لاگ در background thread تا سرعت را کاهش ندهد
        try:
            import threading
            threading.Thread(target=self._write_log, args=(message, response), daemon=True).start()
        except Exception:
            # اگر thread ایجاد نشد، لاگ نکن (سرعت مهم‌تر است)
            pass
    
    def _write_log(self, message: str, response: str):
        """نوشتن لاگ به صورت sync در background"""
        try:
            log_path = CHAT_LOG_PATH
            logs = []
            
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append({
                "instruction": message,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # محدود کردن اندازه لاگ
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # نوشتن بدون indent برای سرعت بیشتر
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False)
        except:
            pass
    
    async def generate_response(
        self,
        message: str,
        session_id: Optional[str] = None,
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
            session_id: شناسه session برای حفظ context
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
        
        # استفاده از حافظه برای ساخت prompt با context
        if session_id:
            prompt = self.memory.get_context_prompt(session_id, message)
        else:
            # اگر session_id نداشتیم، از prompt ساده استفاده کنیم
            prompt = f"User: {message}\nAssistant:"
        
        # Tokenize (کاهش max_length برای سرعت بیشتر)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256  # کاهش از 512 به 256 برای سرعت بیشتر
        ).to(self.peft_model.device)
        
        # Generate با پارامترهای بهینه شده برای سرعت
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_length=5,  # کاهش برای سرعت بیشتر
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=temperature > 0.1,  # فقط اگر temperature بالا باشد sampling کن
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True,
                num_beams=1 if temperature > 0.1 else 1,  # بدون beam search برای سرعت
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
        
        # ذخیره در حافظه اگر session_id داشتیم
        if session_id:
            self.memory.add_context(session_id, message, response)
        
        # لاگ کردن چت برای یادگیری
        self._log_chat(message, response)
        
        return response
