# نصب سریع برای سرورهای ایرانی
# Fast Installation for Iranian Servers

## روش 1: استفاده از اسکریپت Python (پیشنهادی)
## Method 1: Using Python Script (Recommended)

```bash
python install_fast.py
```

## روش 2: استفاده از اسکریپت Bash
## Method 2: Using Bash Script

```bash
chmod +x install_fast.sh
./install_fast.sh
```

## روش 3: نصب دستی با آینه ایرانی
## Method 3: Manual Installation with Iranian Mirror

```bash
# تنظیم آینه
export PIP_INDEX_URL="https://pypi.rasa.ir/simple"
export PIP_TRUSTED_HOST="pypi.rasa.ir"

# یا
export PIP_INDEX_URL="https://pypi.douban.com/simple"
export PIP_TRUSTED_HOST="pypi.douban.com"

# ارتقای pip
pip install --upgrade pip setuptools wheel -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST

# نصب PyTorch (از آینه اصلی)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# نصب وابستگی‌ها
pip install transformers>=4.40.0 accelerate>=0.27.0 peft>=0.8.0 bitsandbytes>=0.43.0 datasets>=2.18.0 -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST

pip install sentencepiece>=0.1.99 protobuf>=3.20.0 scipy>=1.11.0 scikit-learn>=1.3.0 numpy>=1.24.0 pandas>=2.0.0 tqdm>=4.66.0 huggingface-hub>=0.20.0 tokenizers>=0.15.0 safetensors>=0.4.0 -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST
```

## روش 4: نصب یک‌خطی (سریع‌ترین)
## Method 4: One-line Installation (Fastest)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install transformers accelerate peft bitsandbytes datasets sentencepiece protobuf scipy scikit-learn numpy pandas tqdm huggingface-hub tokenizers safetensors -i https://pypi.rasa.ir/simple --trusted-host pypi.rasa.ir
```

## آینه‌های پیشنهادی
## Recommended Mirrors

1. **pypi.rasa.ir** - آینه ایرانی (سریع)
2. **pypi.douban.com** - آینه چینی (سریع)
3. **mirrors.aliyun.com** - آینه علی‌بابا (سریع)

## نکات
## Notes

- PyTorch را از آینه اصلی نصب کنید (سریع‌تر است)
- بقیه پکیج‌ها را از آینه‌های ایرانی نصب کنید
- اگر یک آینه کار نکرد، آینه دیگر را امتحان کنید

