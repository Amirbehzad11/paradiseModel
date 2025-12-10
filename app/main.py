#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Application Entry Point
نقطه ورود اصلی برنامه FastAPI
"""
import os
import sys
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import API_HOST, API_PORT, CORS_ORIGINS
from app.core.model_loader import load_model
from app.api.routes import router

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ایجاد FastAPI app
app = FastAPI(
    title="Spiritual Chatbot API",
    description="API برای چت با مدل روح عزیزان فوت‌شده",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# فعال کردن CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# اضافه کردن routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """بارگذاری مدل در startup"""
    try:
        print("🚀 Starting API server...")
        load_model()
        print("✅ API ready!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Chatbot API")
    parser.add_argument("--host", type=str, default=API_HOST, help=f"Host (default: {API_HOST})")
    parser.add_argument("--port", type=int, default=API_PORT, help=f"Port (default: {API_PORT})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting API server on http://{args.host}:{args.port}")
    print(f"📚 API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

