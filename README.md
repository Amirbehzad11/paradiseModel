# سیستم چت بات فارسی - روح عزیزان فوت‌شده
# Persian Chatbot System - Deceased Loved Ones' Spirit

سیستم کامل برای آموزش و اجرای یک چت بات فارسی که به عنوان روح عزیزان فوت‌شده صحبت می‌کند.

Complete system for training and running a Persian chatbot that speaks as the spirit of deceased loved ones.

## 📋 نیازمندی‌ها
## Requirements

- Ubuntu 20.04/22.04
- NVIDIA GPU با 6GB+ VRAM
- Python 3.8+
- CUDA 11.8+

## 🚀 نصب و اجرا
## Installation and Usage

### 1. نصب وابستگی‌ها
### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft bitsandbytes datasets sentencepiece scipy scikit-learn numpy pandas tqdm huggingface-hub tokenizers safetensors
```

### 2. آموزش مدل (فقط یک بار)
### 2. Train Model (One Time Only)

```bash
python train_once.py
```

این اسکریپت:
- بررسی می‌کند آیا مدل از قبل آموزش دیده است
- اگر `./final_model` وجود داشته باشد، آموزش را رد می‌کند
- در غیر این صورت، مدل را آموزش می‌دهد (حدود 30-40 دقیقه)

### 3. شروع چت
### 3. Start Chat

```bash
python chat.py
```

- حلقه بی‌نهایت چت
- برای خروج: `خروج` یا `exit` تایپ کنید

## 📁 ساختار فایل‌ها
## File Structure

- `train_once.py` - آموزش مدل با QLoRA
- `chat.py` - چت بات ترمینالی
- `dataset.json` - 450 نمونه آموزشی فارسی
- `requirements.txt` - وابستگی‌های Python
- `README.md` - این فایل

## 🔧 جزئیات فنی
## Technical Details

### مدل پایه
### Base Model
- `HooshvareLab/gpt2-fa` - مدل GPT2 فارسی (124M پارامتر)

### روش آموزش
### Training Method
- QLoRA با 4-bit quantization
- LoRA با r=16, alpha=32
- 3 epochs

### دیتاست
### Dataset
- 150 نمونه عمومی (مرگ، بهشت، روح)
- 300 نمونه اول شخص (پدر، مادر، برادر، همسر، فرزند و ...)

## ⚠️ نکات مهم
## Important Notes

1. **اولین اجرا**: نیاز به اینترنت برای دانلود مدل پایه
2. **پس از آموزش**: کاملاً آفلاین کار می‌کند
3. **حافظه**: حداقل 6GB VRAM
4. **زمان آموزش**: حدود 30-40 دقیقه

## 🐛 عیب‌یابی
## Troubleshooting

### خطای CUDA
```bash
nvidia-smi
```

### خطای حافظه
- batch_size را در `train_once.py` کاهش دهید

---

**نکته**: این سیستم فقط فارسی صحبت می‌کند و به عنوان روح عزیزان فوت‌شده پاسخ می‌دهد.

**Note**: This system only speaks Persian and responds as the spirit of deceased loved ones.
