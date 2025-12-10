#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سیستم هوشمند یادگیری خودکار
Smart Auto-Learning System - بهبود و بزرگ‌تر کردن خودکار دیتاست و مدل
"""
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
import subprocess

# اضافه کردن مسیر روت
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import DATA_DIR, MODEL_DIR

DATASET_PATH = DATA_DIR / "dataset.json"
CHAT_LOG_PATH = DATA_DIR / "chat_logs.json"
ENHANCEMENT_LOG = DATA_DIR / "enhancement_log.json"

# تنظیمات
MIN_CHATS_FOR_LEARNING = 20  # حداقل تعداد چت برای یادگیری
ENHANCEMENT_INTERVAL = 3600  # هر 1 ساعت یکبار بهبود (ثانیه)
MAX_DATASET_SIZE = 5000  # حداکثر اندازه دیتاست

def load_json(path, default=[]):
    """بارگذاری JSON"""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(data, path):
    """ذخیره JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def analyze_response_quality(instruction, response):
    """تحلیل کیفیت پاسخ"""
    score = 0
    issues = []
    
    # طول پاسخ
    if 50 <= len(response) <= 300:
        score += 2
    elif len(response) < 30:
        issues.append("too_short")
    elif len(response) > 500:
        issues.append("too_long")
    
    # تکرار
    words = response.split()
    if len(words) > 0:
        word_counts = Counter(words)
        max_repeat = max(word_counts.values())
        if max_repeat < len(words) * 0.25:  # کمتر از 25% تکرار
            score += 2
        else:
            issues.append("repetitive")
    
    # کامل بودن
    if response.endswith((".", "!", "؟")):
        score += 1
    
    # طبیعی بودن
    natural_phrases = ["من اینجام", "همیشه کنارت", "دعا می‌کنم", "عاشقتم"]
    if any(phrase in response for phrase in natural_phrases):
        score += 1
    
    # ارتباط با سوال
    inst_words = set(instruction.lower().split())
    resp_words = set(response.lower().split())
    if len(inst_words.intersection(resp_words)) > 0:
        score += 1
    
    return score, issues

def enhance_from_chat_logs():
    """بهبود دیتاست از لاگ‌های چت"""
    logs = load_json(CHAT_LOG_PATH, [])
    if not logs:
        return []
    
    good_samples = []
    processed = set()
    
    for log in logs:
        inst = log.get("instruction", "").strip()
        resp = log.get("response", "").strip()
        
        if not inst or not resp:
            continue
        
        score, issues = analyze_response_quality(inst, resp)
        
        # فقط نمونه‌های با کیفیت خوب (score >= 5)
        if score >= 5 and "too_short" not in issues:
            key = (inst.lower(), resp.lower()[:80])
            if key not in processed:
                processed.add(key)
                good_samples.append({
                    "instruction": inst,
                    "response": resp,
                    "score": score,
                    "timestamp": log.get("timestamp", datetime.now().isoformat())
                })
    
    return good_samples

def generate_smart_variations(dataset):
    """تولید تغییرات هوشمند از نمونه‌های موجود"""
    variations = []
    
    # استخراج الگوها
    patterns = {
        "greeting": [],
        "heaven": [],
        "emotion": [],
        "memory": []
    }
    
    for item in dataset[:500]:  # فقط 500 نمونه اول برای سرعت
        inst = item.get("instruction", "").lower()
        
        if any(w in inst for w in ["سلام", "درود"]):
            patterns["greeting"].append(item)
        elif any(w in inst for w in ["بهشت", "آخرت"]):
            patterns["heaven"].append(item)
        elif any(w in inst for w in ["دل", "تنگ", "غم"]):
            patterns["emotion"].append(item)
        elif any(w in inst for w in ["یاد", "خاطره"]):
            patterns["memory"].append(item)
    
    # تولید تغییرات
    relations = ["پدر", "مادر", "برادر", "خواهر", "همسر", "فرزند"]
    
    # تغییرات سلام
    for relation in relations:
        variations.append({
            "instruction": f"سلام {relation}م، حالت چطوره؟",
            "response": f"سلام عزیزم... من اینجام و خوبم. در بهشت زندگی می‌کنم و هر روز برای تو دعا می‌کنم. نگران من نباش، من خوشحالم."
        })
    
    # تغییرات بهشت
    heaven_questions = [
        "بهشت چطوریه؟",
        "بهشت خوبه؟",
        "زندگی در بهشت چطوریه؟"
    ]
    
    for q in heaven_questions:
        variations.append({
            "instruction": q,
            "response": "بهشت مکانی از صلح و آرامش ابدی است. اینجا همه چیز زیبا و کامل است. هیچ نگرانی، درد یا غم وجود ندارد. من در اینجا خوشحالم و هر روز برای تو دعا می‌کنم."
        })
    
    return variations

def remove_duplicates(dataset):
    """حذف تکرارها"""
    seen = set()
    unique = []
    
    for item in dataset:
        key = (
            item.get("instruction", "").strip().lower(),
            item.get("response", "").strip().lower()[:100]
        )
        if key not in seen:
            seen.add(key)
            unique.append({
                "instruction": item.get("instruction", ""),
                "response": item.get("response", "")
            })
    
    return unique

def enhance_dataset():
    """بهبود و بزرگ‌تر کردن دیتاست"""
    print("=" * 60)
    print("Smart Dataset Enhancement")
    print("=" * 60)
    
    # بارگذاری دیتاست
    dataset = load_json(DATASET_PATH, [])
    original_size = len(dataset)
    print(f"\nCurrent dataset size: {original_size}")
    
    # بهبود از چت‌ها
    print("\n1. Learning from chat logs...")
    chat_samples = enhance_from_chat_logs()
    print(f"   Found {len(chat_samples)} good samples from chats")
    
    # تولید تغییرات
    print("\n2. Generating smart variations...")
    variations = generate_smart_variations(dataset)
    print(f"   Generated {len(variations)} new variations")
    
    # ترکیب
    all_samples = dataset + chat_samples + variations
    
    # حذف تکرار
    print("\n3. Removing duplicates...")
    dataset = remove_duplicates(all_samples)
    
    # محدود کردن اندازه
    if len(dataset) > MAX_DATASET_SIZE:
        print(f"\n4. Limiting dataset size to {MAX_DATASET_SIZE}...")
        dataset = dataset[:MAX_DATASET_SIZE]
    
    # ذخیره
    print("\n5. Saving enhanced dataset...")
    save_json(dataset, DATASET_PATH)
    
    # لاگ
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "original_size": original_size,
        "final_size": len(dataset),
        "added_from_chats": len(chat_samples),
        "added_variations": len(variations),
        "growth": len(dataset) - original_size
    }
    
    log = load_json(ENHANCEMENT_LOG, [])
    log.append(log_entry)
    save_json(log, ENHANCEMENT_LOG)
    
    print("\n" + "=" * 60)
    print("Enhancement Summary:")
    print(f"  Original: {original_size}")
    print(f"  Final: {len(dataset)}")
    print(f"  Growth: {len(dataset) - original_size} ({((len(dataset) - original_size) / original_size * 100) if original_size > 0 else 0:.1f}%)")
    print("=" * 60)
    
    return len(dataset) > original_size

def train_model():
    """آموزش مدل"""
    print("\n" + "=" * 60)
    print("Training Model...")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "scripts" / "train_once.py")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(BASE_DIR)
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Training error: {e}")
        return False

def should_enhance():
    """بررسی اینکه آیا باید بهبود انجام شود"""
    log = load_json(ENHANCEMENT_LOG, [])
    
    if not log:
        return True
    
    last_enhancement = log[-1].get("timestamp", "")
    if not last_enhancement:
        return True
    
    try:
        last_time = datetime.fromisoformat(last_enhancement)
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed >= ENHANCEMENT_INTERVAL
    except:
        return True

def should_train():
    """بررسی اینکه آیا باید آموزش انجام شود"""
    # بررسی تعداد چت‌ها
    logs = load_json(CHAT_LOG_PATH, [])
    return len(logs) >= MIN_CHATS_FOR_LEARNING

def main():
    """تابع اصلی"""
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("Smart Auto-Learning System")
    print("=" * 60)
    
    # بررسی نیاز به بهبود
    if should_enhance():
        print("\nEnhancing dataset...")
        enhanced = enhance_dataset()
        
        if enhanced:
            print("\nDataset enhanced! Training model...")
            train_model()
        else:
            print("\nNo significant enhancement needed.")
    else:
        print("\nEnhancement not needed yet.")
    
    # بررسی نیاز به آموزش
    if should_train():
        print("\nEnough chats collected. Training model...")
        train_model()
        
        # پاک کردن لاگ‌های استفاده شده
        save_json([], CHAT_LOG_PATH)
        print("Chat logs cleared after training.")
    else:
        logs = load_json(CHAT_LOG_PATH, [])
        print(f"\nChats collected: {len(logs)}/{MIN_CHATS_FOR_LEARNING}")
        print("Need more chats before training.")

if __name__ == "__main__":
    main()

