# راهنمای سریع - RTX 3080 Fine-tuning
# Quick Start Guide - RTX 3080 Fine-tuning

## 🚀 شروع سریع

### 1. نصب PyTorch با CUDA

```bash
# برای CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# برای CUDA 12.1  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. نصب وابستگی‌ها

```bash
pip install -r requirements_3080.txt
```

### 3. آماده‌سازی دیتاست

```bash
python scripts/dataset_prep.py
```

این دستور:
- دیتاست را از `data/dataset.json` می‌خواند
- به فرمت ChatML تبدیل می‌کند
- در `data/data_persian.json` ذخیره می‌کند

### 4. فاین‌تیون

```bash
python scripts/train_3080.py
```

**زمان**: حدود 2-5 ساعت (بسته به تعداد نمونه‌ها)

### 5. تست

```bash
python scripts/inference_3080.py
```

سپس مرورگر را باز کنید: `http://localhost:7860`

## 📋 چک‌لیست

- [ ] PyTorch با CUDA نصب شده
- [ ] وابستگی‌ها نصب شده‌اند (`pip install -r requirements_3080.txt`)
- [ ] دیتاست آماده است (`data/data_persian.json`)
- [ ] حداقل 500 نمونه در دیتاست
- [ ] GPU در دسترس است

## ⚙️ تنظیمات

### برای RTX 3080 (پیش‌فرض - بهینه):

```python
LORA_R = 64
LORA_ALPHA = 16
BATCH_SIZE = 3
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
```

### اگر حافظه کم دارید:

```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
MAX_SEQ_LENGTH = 1024
```

## 📊 فرمت دیتاست

دیتاست باید به این فرمت باشد:

```json
[
  {
    "messages": [
      {"role": "user", "content": "سلام مامان"},
      {"role": "assistant", "content": "سلام عزیزم... من اینجام..."}
    ]
  }
]
```

## 🔍 عیب‌یابی

### خطا: "CUDA out of memory"
- `BATCH_SIZE` را کاهش دهید (3 → 2 → 1)

### خطا: "Model not found"
- Hugging Face token را تنظیم کنید:
```bash
huggingface-cli login
```

## 📚 مستندات کامل

برای جزئیات بیشتر: `docs/RTX3080_SETUP.md`

## ✅ نتیجه

بعد از فاین‌تیون:
- مدل در `models/llama3_8b_finetuned` ذخیره می‌شود
- می‌توانید با `inference_3080.py` تست کنید
- پاسخ‌ها طبیعی‌تر و احساسی‌تر می‌شوند

