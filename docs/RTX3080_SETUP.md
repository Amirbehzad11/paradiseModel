# راهنمای کامل فاین‌تیون برای RTX 3080
# Complete Fine-tuning Guide for RTX 3080

## 🎯 هدف

فاین‌تیون یک مدل 8B فارسی/معنوی روی RTX 3080 (10GB) با QLoRA برای:
- ✅ چت‌بات طبیعی‌تر و احساسی‌تر
- ✅ شبیه‌سازی دقیق مادر/عزیزان فوت‌شده
- ✅ استفاده بهینه از حافظه GPU

## 📋 پیش‌نیازها

### سخت‌افزار:
- RTX 3080 10GB (یا GPU مشابه)
- حداقل 16GB RAM
- فضای دیسک: 20GB+

### نرم‌افزار:
- Python 3.10+
- CUDA 11.8 یا 12.1
- PyTorch با CUDA

## 🚀 نصب

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

### 3. نصب Flash Attention (اختیاری - برای سرعت بیشتر)

```bash
# نیاز به CUDA toolkit دارد
pip install flash-attn --no-build-isolation
```

## 📚 آماده‌سازی دیتاست

### 1. تبدیل دیتاست به فرمت ChatML

```bash
python scripts/dataset_prep.py
```

این اسکریپت:
- دیتاست را از `data/dataset.json` می‌خواند
- به فرمت ChatML تبدیل می‌کند
- در `data/data_persian.json` ذخیره می‌کند

### 2. فرمت دیتاست

دیتاست باید به این فرمت باشد:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "مامان جون، امروز خیلی دلم گرفته..."
      },
      {
        "role": "assistant",
        "content": "عزیز دلم، بیا پیش مامان. تو که می‌دونی هیچ‌وقت تنهات نمی‌ذارم. بگو چی شده جوجه‌ام؟"
      }
    ]
  }
]
```

### 3. بررسی دیتاست

```bash
# بررسی تعداد نمونه‌ها
python -c "import json; data = json.load(open('data/data_persian.json', 'r', encoding='utf-8')); print(f'Total: {len(data)} examples')"
```

**توصیه**: حداقل 500-2000 نمونه برای نتیجه خوب

## 🎓 فاین‌تیون

### اجرای فاین‌تیون

```bash
python scripts/train_3080.py
```

### تنظیمات بهینه برای RTX 3080

در `scripts/train_3080.py`:

```python
# LoRA
LORA_R = 64              # رتبه LoRA (بالاتر = کیفیت بهتر)
LORA_ALPHA = 16          # Alpha برای LoRA
LORA_DROPOUT = 0.1       # Dropout

# Batch
BATCH_SIZE = 3           # per device
GRADIENT_ACCUMULATION = 4 # effective batch = 12

# Training
LEARNING_RATE = 2e-4     # Learning rate
NUM_EPOCHS = 5           # تعداد epoch
MAX_SEQ_LENGTH = 2048    # حداکثر طول sequence
```

### زمان آموزش

- با 1000 نمونه: حدود 2-3 ساعت
- با 2000 نمونه: حدود 4-5 ساعت

### نظارت بر آموزش

لاگ‌ها در `checkpoints_3080/` ذخیره می‌شوند. می‌توانید loss را بررسی کنید:

```bash
# مشاهده لاگ‌ها
tail -f checkpoints_3080/training.log
```

## 🧪 تست و Inference

### 1. با Gradio (پیشنهادی)

```bash
python scripts/inference_3080.py
```

سپس مرورگر را باز کنید: `http://localhost:7860`

### 2. با Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_PATH = "models/llama3_8b_finetuned"

# بارگذاری
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# چت
prompt = "<|user|>\nسلام مامان<|end|>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## ⚙️ تنظیمات پیشرفته

### افزایش کیفیت (اگر حافظه کافی دارید)

```python
LORA_R = 128  # از 64 به 128
LORA_ALPHA = 32  # از 16 به 32
BATCH_SIZE = 2  # کاهش batch برای جا دادن
```

### افزایش سرعت

```python
BATCH_SIZE = 4  # افزایش batch
GRADIENT_ACCUMULATION = 2  # کاهش accumulation
MAX_SEQ_LENGTH = 1024  # کاهش طول sequence
```

### کاهش حافظه

```python
BATCH_SIZE = 1  # کاهش batch
GRADIENT_ACCUMULATION = 8  # افزایش accumulation
MAX_SEQ_LENGTH = 512  # کاهش طول sequence
```

## 🔍 عیب‌یابی

### خطا: "CUDA out of memory"

**راه‌حل:**
1. `BATCH_SIZE` را کاهش دهید (3 → 2 → 1)
2. `MAX_SEQ_LENGTH` را کاهش دهید (2048 → 1024)
3. `gradient_checkpointing=True` را فعال کنید (فعال است)

### خطا: "Model not found"

**راه‌حل:**
```bash
# دانلود مدل از Hugging Face
huggingface-cli login
# سپس دوباره train کنید
```

### خطا: "Flash Attention not available"

**راه‌حل:**
- این خطا مهم نیست، مدل با `eager` attention هم کار می‌کند
- اگر می‌خواهید Flash Attention داشته باشید، CUDA toolkit را نصب کنید

## 📊 بررسی نتایج

### معیارهای کیفیت:

1. **Loss**: باید کاهش یابد (مثلاً از 2.5 به 1.5)
2. **Perplexity**: باید کاهش یابد
3. **پاسخ‌ها**: باید طبیعی‌تر و احساسی‌تر باشند

### تست دستی:

چند سوال بپرسید و بررسی کنید:
- ✅ پاسخ‌ها طبیعی هستند؟
- ✅ احساسی و شخصی هستند؟
- ✅ شبیه مادر/عزیز فوت‌شده هستند؟

## 💡 نکات مهم

1. **کیفیت دیتاست**: مهم‌تر از تعداد است
2. **صبر**: فاین‌تیون زمان می‌برد (2-5 ساعت)
3. **نظارت**: loss را بررسی کنید
4. **تست**: بعد از آموزش حتماً تست کنید

## 📝 خلاصه دستورات

```bash
# 1. آماده‌سازی دیتاست
python scripts/dataset_prep.py

# 2. فاین‌تیون
python scripts/train_3080.py

# 3. تست
python scripts/inference_3080.py
```

## ✅ چک‌لیست

- [ ] PyTorch با CUDA نصب شده
- [ ] وابستگی‌ها نصب شده‌اند
- [ ] دیتاست آماده است (data_persian.json)
- [ ] حداقل 500 نمونه در دیتاست
- [ ] GPU در دسترس است
- [ ] فضای کافی برای ذخیره مدل

## 🎯 نتیجه

بعد از فاین‌تیون:
- ✅ مدل طبیعی‌تر صحبت می‌کند
- ✅ احساسی‌تر و شخصی‌تر است
- ✅ شبیه‌سازی دقیق‌تر مادر/عزیز فوت‌شده
- ✅ استفاده بهینه از RTX 3080

