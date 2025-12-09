# گزینه‌های مدل - کاملاً باز بدون نیاز به مجوز
# Model Options - Fully Open, No License Required

## مدل‌های پیشنهادی
## Recommended Models

### 1. Microsoft Phi-3-mini (پیشنهادی ⭐)
**بهترین انتخاب برای instruction following**

```python
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
```

**مزایا:**
- ✅ کاملاً باز، بدون نیاز به مجوز
- ✅ عالی برای instruction following
- ✅ حافظه کم (حدود 4-6GB VRAM)
- ✅ کیفیت بالا
- ✅ پشتیبانی از فارسی و انگلیسی

**معایب:**
- ⚠️ کمی کوچکتر از Llama-3.2

---

### 2. Qwen2-1.5B-Instruct
**عالی برای فارسی/انگلیسی**

```python
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
```

**مزایا:**
- ✅ کاملاً باز
- ✅ پشتیبانی عالی از فارسی
- ✅ حافظه کم (حدود 4-6GB VRAM)
- ✅ کیفیت خوب

---

### 3. DialoGPT-medium (برای dialogue)
**مناسب برای گفتگوهای ساده**

```python
BASE_MODEL = "microsoft/DialoGPT-medium"
```

**مزایا:**
- ✅ کاملاً باز
- ✅ حافظه خیلی کم (حدود 2-3GB VRAM)
- ✅ مناسب برای dialogue

**معایب:**
- ⚠️ نیاز به فرمت خاص برای instruction
- ⚠️ کیفیت کمتر از Phi-3

---

### 4. GPT2-medium (برای text generation)
**مناسب برای تولید متن**

```python
BASE_MODEL = "gpt2-medium"
```

**مزایا:**
- ✅ کاملاً باز
- ✅ حافظه کم (حدود 2-3GB VRAM)
- ✅ سریع

**معایب:**
- ⚠️ برای instruction following مناسب نیست
- ⚠️ نیاز به فرمت خاص

---

## نحوه تغییر مدل
## How to Change Model

در دو فایل `train_once.py` و `chat.py`، خط زیر را تغییر دهید:

```python
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # مدل مورد نظر
```

**مهم:** هر دو فایل باید یک مدل داشته باشند!

---

## مقایسه سریع
## Quick Comparison

| مدل | VRAM | کیفیت | فارسی | Instruction |
|-----|------|-------|-------|-------------|
| Phi-3-mini | 4-6GB | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ |
| Qwen2-1.5B | 4-6GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| DialoGPT | 2-3GB | ⭐⭐⭐ | ⚠️ | ⭐⭐ |
| GPT2 | 2-3GB | ⭐⭐ | ⚠️ | ⭐ |

---

## پیشنهاد نهایی
## Final Recommendation

**برای بهترین نتیجه:** `microsoft/Phi-3-mini-4k-instruct`

این مدل:
- کاملاً باز است (بدون نیاز به مجوز)
- عالی برای instruction following
- حافظه کمی می‌گیرد
- کیفیت بالا دارد

