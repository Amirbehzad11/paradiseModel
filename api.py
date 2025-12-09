#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API برای استفاده خارجی از مدل چت
REST API for external use of chat model
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ایجاد FastAPI app
app = FastAPI(
    title="Spiritual Chatbot API",
    description="API برای چت با مدل روح عزیزان فوت‌شده",
    version="1.0.0"
)

# فعال کردن CORS برای استفاده از frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در production بهتر است محدود شود
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مدل‌های global
peft_model = None
tokenizer = None
BASE_MODEL = "HooshvareLab/gpt2-fa"

# Schema برای request و response
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

def load_model():
    """بارگذاری مدل"""
    global peft_model, tokenizer
    
    if not os.path.exists("./final_model"):
        raise FileNotFoundError("Model not found. Please run train_once.py first.")
    
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
    
    print("✅ Model loaded successfully!")

# بارگذاری مدل در startup
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

@app.get("/", response_model=HealthResponse)
async def root():
    """بررسی وضعیت API"""
    return {
        "status": "running",
        "model_loaded": peft_model is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """بررسی سلامت API"""
    return {
        "status": "healthy" if peft_model is not None else "model_not_loaded",
        "model_loaded": peft_model is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    چت با مدل
    
    Args:
        request: درخواست چت شامل message و پارامترهای generation
    
    Returns:
        پاسخ مدل
    """
    if peft_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # ساخت prompt
        prompt = f"User: {request.message}\nAssistant:"
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(peft_model.device)
        
        # Generate
        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # Decode response
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        # حذف prompt اگر در response باشد
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return ChatResponse(response=response, status="success")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat/simple")
async def chat_simple(message: str):
    """
    چت ساده (فقط message)
    
    Args:
        message: متن کاربر
    
    Returns:
        پاسخ مدل
    """
    request = ChatRequest(message=message)
    return await chat(request)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Chatbot API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting API server on http://{args.host}:{args.port}")
    print(f"📚 API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

