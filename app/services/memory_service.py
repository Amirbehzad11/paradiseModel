#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سرویس حافظه - مدیریت context و نقش‌ها
Memory Service - Context and Role Management
"""
import re
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json
from pathlib import Path
from app.core.config import DATA_DIR

# نقش‌های شناخته شده
ROLES = {
    "مادر": ["مادر", "مامان", "مامانی", "مادرت", "مادرت", "مادرت"],
    "پدر": ["پدر", "بابا", "بابایی", "پدرت", "پدرت", "پدرت"],
    "برادر": ["برادر", "داداش", "برادرت", "برادرت"],
    "خواهر": ["خواهر", "خواهرم", "خواهرت", "خواهرت"],
    "همسر": ["همسر", "عشق", "عشقم", "همسرم", "همسرت"],
    "فرزند": ["فرزند", "بچه", "بچم", "فرزندت", "فرزندت"]
}

# فایل ذخیره session ها
SESSIONS_PATH = DATA_DIR / "sessions.json"


class MemoryService:
    """سرویس حافظه برای نگه‌داری context"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self._load_sessions()
    
    def _load_sessions(self):
        """بارگذاری session ها از فایل"""
        try:
            if SESSIONS_PATH.exists():
                with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # فقط session های فعال (کمتر از 24 ساعت) را نگه دار
                    now = datetime.now()
                    for session_id, session_data in data.items():
                        last_access = datetime.fromisoformat(session_data.get("last_access", now.isoformat()))
                        if (now - last_access).total_seconds() < 86400:  # 24 ساعت
                            self.sessions[session_id] = session_data
        except:
            self.sessions = {}
    
    def _save_sessions(self):
        """ذخیره session ها در فایل"""
        try:
            SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, ensure_ascii=False)
        except:
            pass
    
    def extract_role(self, message: str) -> Optional[str]:
        """استخراج نقش از پیام"""
        message_lower = message.lower()
        
        # جستجوی نقش‌ها
        for role, keywords in ROLES.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return role
        
        return None
    
    def get_or_create_session(self, session_id: str) -> Dict:
        """دریافت یا ایجاد session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "role": None,
                "context": [],
                "created_at": datetime.now().isoformat(),
                "last_access": datetime.now().isoformat()
            }
        else:
            self.sessions[session_id]["last_access"] = datetime.now().isoformat()
        
        self._save_sessions()
        return self.sessions[session_id]
    
    def update_role(self, session_id: str, role: str):
        """به‌روزرسانی نقش در session"""
        session = self.get_or_create_session(session_id)
        if role and role in ROLES:
            session["role"] = role
            self._save_sessions()
    
    def add_context(self, session_id: str, message: str, response: str):
        """افزودن context به session"""
        session = self.get_or_create_session(session_id)
        session["context"].append({
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # محدود کردن اندازه context (آخرین 10 پیام)
        if len(session["context"]) > 10:
            session["context"] = session["context"][-10:]
        
        self._save_sessions()
    
    def get_context_prompt(self, session_id: str, message: str) -> str:
        """ساخت prompt با context"""
        session = self.get_or_create_session(session_id)
        
        # استخراج نقش از پیام
        role = self.extract_role(message)
        if role:
            session["role"] = role
        
        # ساخت prompt بر اساس نقش
        role_context = ""
        if session["role"]:
            role_name = session["role"]
            role_context = f"تو {role_name} این کاربر هستی. به عنوان {role_name} با او صحبت کن. "
        
        # اضافه کردن context قبلی
        context_history = ""
        if session["context"]:
            recent_context = session["context"][-3:]  # آخرین 3 پیام
            context_parts = []
            for ctx in recent_context:
                context_parts.append(f"User: {ctx['message']}\nAssistant: {ctx['response']}")
            context_history = "\n".join(context_parts) + "\n"
        
        # ساخت prompt نهایی
        prompt = f"{role_context}{context_history}User: {message}\nAssistant:"
        
        return prompt
    
    def clear_session(self, session_id: str):
        """پاک کردن session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
    
    def cleanup_old_sessions(self):
        """پاک کردن session های قدیمی (بیش از 24 ساعت)"""
        now = datetime.now()
        to_remove = []
        
        for session_id, session_data in self.sessions.items():
            last_access = datetime.fromisoformat(session_data.get("last_access", now.isoformat()))
            if (now - last_access).total_seconds() >= 86400:  # 24 ساعت
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
        
        if to_remove:
            self._save_sessions()

