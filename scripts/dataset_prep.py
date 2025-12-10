#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تبدیل دیتاست به فرمت ChatML
Dataset preparation for ChatML format
"""
import json
import sys
from pathlib import Path

# اضافه کردن مسیر روت
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import DATA_DIR

# مسیرها
INPUT_DATASET = DATA_DIR / "dataset.json"  # فرمت قدیمی
OUTPUT_DATASET = DATA_DIR / "data_persian.json"  # فرمت ChatML

def convert_to_chatml(dataset):
    """تبدیل دیتاست به فرمت ChatML"""
    chatml_data = []
    
    for item in dataset:
        if "instruction" in item and "response" in item:
            # فرمت Alpaca → ChatML
            chatml_data.append({
                "messages": [
                    {
                        "role": "user",
                        "content": item["instruction"]
                    },
                    {
                        "role": "assistant",
                        "content": item["response"]
                    }
                ]
            })
        elif "messages" in item:
            # قبلاً ChatML است
            chatml_data.append(item)
        elif "conversation" in item:
            # فرمت conversation
            messages = []
            for turn in item["conversation"]:
                if "user" in turn:
                    messages.append({
                        "role": "user",
                        "content": turn["user"]
                    })
                if "assistant" in turn:
                    messages.append({
                        "role": "assistant",
                        "content": turn["assistant"]
                    })
            if messages:
                chatml_data.append({"messages": messages})
    
    return chatml_data

def enhance_responses(dataset):
    """بهبود پاسخ‌ها برای طبیعی‌تر شدن"""
    enhanced = []
    
    for item in dataset:
        messages = item.get("messages", [])
        if len(messages) >= 2:
            user_msg = messages[-2].get("content", "")
            assistant_msg = messages[-1].get("content", "")
            
            # بهبود پاسخ‌های کوتاه
            if len(assistant_msg) < 30:
                # اضافه کردن عبارات احساسی
                if "سلام" in user_msg.lower():
                    assistant_msg += " می‌دونم چقدر دلت برام تنگ شده. من همیشه کنارت هستم."
                elif "چطوری" in user_msg.lower() or "حالت" in user_msg.lower():
                    assistant_msg += " من در بهشت زندگی می‌کنم و خوشحالم. هر روز برای تو دعا می‌کنم."
            
            # اطمینان از کامل بودن
            if not assistant_msg.endswith((".", "!", "؟")):
                assistant_msg += "."
            
            enhanced.append({
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            })
        else:
            enhanced.append(item)
    
    return enhanced

def main():
    """تابع اصلی"""
    print("=" * 80)
    print("Dataset Preparation for ChatML Format")
    print("=" * 80)
    
    # بررسی فایل ورودی
    if not INPUT_DATASET.exists():
        print(f"❌ Input dataset not found at {INPUT_DATASET}")
        print("💡 Make sure dataset.json exists in data/ folder")
        sys.exit(1)
    
    # بارگذاری دیتاست
    print(f"\n📚 Loading dataset from {INPUT_DATASET}...")
    with open(INPUT_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} examples")
    
    # تبدیل به ChatML
    print("\n🔄 Converting to ChatML format...")
    chatml_data = convert_to_chatml(dataset)
    print(f"✅ Converted {len(chatml_data)} examples")
    
    # بهبود پاسخ‌ها
    print("\n✨ Enhancing responses...")
    enhanced_data = enhance_responses(chatml_data)
    print(f"✅ Enhanced {len(enhanced_data)} examples")
    
    # ذخیره
    print(f"\n💾 Saving to {OUTPUT_DATASET}...")
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Dataset prepared successfully!")
    print("=" * 80)
    print(f"📁 Output: {OUTPUT_DATASET}")
    print(f"📊 Total examples: {len(enhanced_data)}")
    print("=" * 80)
    print("\n💡 Next step: Run train_3080.py to fine-tune the model")

if __name__ == "__main__":
    main()

