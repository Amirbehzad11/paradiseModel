#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست API
Test API
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """تست health endpoint"""
    print("🔍 Testing /health...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    print()

def test_chat(message):
    """تست chat endpoint"""
    print(f"💬 Testing /chat with message: {message}")
    
    data = {
        "message": message,
        "max_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2
    }
    
    response = requests.post(f"{API_URL}/chat", json=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_chat_simple(message):
    """تست chat/simple endpoint"""
    print(f"💬 Testing /chat/simple with message: {message}")
    
    response = requests.post(
        f"{API_URL}/chat/simple",
        params={"message": message}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Chatbot API")
    print("=" * 60)
    print()
    
    # تست health
    try:
        test_health()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 مطمئن شوید API در حال اجرا است: python run.py یا python -m app.main")
        exit(1)
    
    # تست chat
    test_chat("سلام خوبی؟")
    
    # تست chat simple
    test_chat_simple("این عکس پدرمه که سال ۹۸ فوت کرد")
    
    print("✅ Tests completed!")

