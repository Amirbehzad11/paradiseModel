# راهنمای دسترسی به مدل Llama
# Llama Model Access Guide

## مشکل: خطای 403 - دسترسی محدود
## Problem: 403 Error - Restricted Access

اگر این خطا را می‌بینید:
If you see this error:

```
403 Client Error. Cannot access gated repo
Access to model meta-llama/Llama-3.2-1B-Instruct is restricted
```

## راه حل (3 مرحله)
## Solution (3 Steps)

### مرحله 1: پذیرش مجوز در Hugging Face
### Step 1: Accept License on Hugging Face

1. به این آدرس بروید:
   Visit this URL:
   ```
   https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   ```

2. روی دکمه **"Agree and access repository"** کلیک کنید
   Click the **"Agree and access repository"** button

3. اگر لاگین نیستید، ابتدا لاگین کنید
   If not logged in, login first

### مرحله 2: لاگین در ترمینال
### Step 2: Login in Terminal

```bash
huggingface-cli login
```

توکن خود را وارد کنید (از https://huggingface.co/settings/tokens)
Enter your token (from https://huggingface.co/settings/tokens)

### مرحله 3: صبر برای فعال شدن دسترسی
### Step 3: Wait for Access Activation

- معمولاً 1-5 دقیقه طول می‌کشد
- Usually takes 1-5 minutes
- سپس دوباره `python train_once.py` را اجرا کنید
- Then run `python train_once.py` again

## بررسی دسترسی
## Check Access

برای بررسی اینکه دسترسی فعال شده است:

```bash
# تست دسترسی
python -c "from huggingface_hub import HfApi; api = HfApi(); print('Access:', api.model_info('meta-llama/Llama-3.2-1B-Instruct'))"
```

## راه حل جایگزین: استفاده از مدل دیگر
## Alternative: Use Different Model

اگر نمی‌خواهید منتظر بمانید، می‌توانید از مدل‌های دیگر استفاده کنید:

```python
# در train_once.py و chat.py تغییر دهید:
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # نیازی به مجوز ندارد
# یا
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"  # نیازی به مجوز ندارد
```

## نکات مهم
## Important Notes

- ✅ مجوز را فقط یک بار باید بپذیرید
- ✅ You only need to accept the license once
- ✅ توکن باید معتبر باشد
- ✅ Token must be valid
- ✅ دسترسی ممکن است چند دقیقه طول بکشد
- ✅ Access may take a few minutes

