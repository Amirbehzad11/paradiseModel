#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """درخواست چت"""
    message: str = Field(..., description="پیام کاربر")
    max_tokens: int = Field(300, ge=1, le=1000, description="حداکثر تعداد token")
    temperature: float = Field(0.9, ge=0.1, le=2.0, description="کنترل خلاقیت")
    top_p: float = Field(0.95, ge=0.1, le=1.0, description="Nucleus sampling")
    top_k: int = Field(50, ge=1, le=100, description="تعداد کلمات برتر")
    repetition_penalty: float = Field(1.4, ge=1.0, le=2.0, description="جلوگیری از تکرار")
    no_repeat_ngram_size: int = Field(3, ge=1, le=5, description="جلوگیری از تکرار n-gram")


class ChatResponse(BaseModel):
    """پاسخ چت"""
    response: str = Field(..., description="پاسخ مدل")
    status: str = Field("success", description="وضعیت پاسخ")


class HealthResponse(BaseModel):
    """پاسخ وضعیت API"""
    status: str = Field(..., description="وضعیت API")
    model_loaded: bool = Field(..., description="آیا مدل بارگذاری شده است")

