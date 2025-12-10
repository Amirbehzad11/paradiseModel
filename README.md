# Chatbot API - API چت بات

یک API کامل و حرفه‌ای برای چت با مدل زبان فارسی با استفاده از FastAPI و PEFT.

## 📁 ساختار پروژه

```
TTS_MODEL/
├── app/                    # کد اصلی برنامه
│   ├── api/                # API endpoints
│   │   ├── models.py       # Pydantic models
│   │   └── routes.py       # API routes
│   ├── core/               # Core functionality
│   │   ├── config.py       # تنظیمات
│   │   └── model_loader.py # بارگذاری مدل
│   ├── services/           # Business logic
│   │   └── chat_service.py # سرویس چت
│   └── main.py             # FastAPI app
├── scripts/                 # اسکریپت‌های کمکی
│   ├── train_once.py       # آموزش یکباره
│   ├── train_incremental.py # آموزش تدریجی
│   ├── continuous_train.py # آموزش پیوسته
│   └── chat.py             # چت تعاملی
├── tests/                  # تست‌ها
│   ├── test_api.py
│   └── test_model.py
├── docs/                   # مستندات
├── data/                   # داده‌ها
│   └── dataset.json
├── models/                 # مدل‌های ذخیره شده
│   └── final_model/
├── logs/                   # لاگ‌ها
├── requirements.txt
└── README.md
```

## 🚀 نصب و راه‌اندازی

### 1. نصب وابستگی‌ها

```bash
pip install -r requirements.txt
```

**نکته:** PyTorch باید جداگانه نصب شود با CUDA:

```bash
# برای CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# برای CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. آماده‌سازی داده‌ها

فایل `data/dataset.json` را با داده‌های خود پر کنید:

```json
[
  {
    "instruction": "سلام خوبی؟",
    "response": "سلام عزیزم، من اینجام..."
  }
]
```

### 3. آموزش مدل

```bash
python scripts/train_once.py
```

### 4. اجرای API

```bash
python -m app.main
```

یا:

```bash
python -m app.main --host 0.0.0.0 --port 8000
```

API روی `http://localhost:8000` اجرا می‌شود.

## 📡 استفاده از API

### مستندات خودکار

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

#### 1. Health Check

```bash
GET /health
```

#### 2. Chat

```bash
POST /chat
Content-Type: application/json

{
  "message": "سلام خوبی؟",
  "max_tokens": 300,
  "temperature": 0.9,
  "top_p": 0.95,
  "top_k": 50,
  "repetition_penalty": 1.4,
  "no_repeat_ngram_size": 3
}
```

#### 3. Chat Simple

```bash
POST /chat/simple?message=سلام
```

#### 4. Gradio Compatible

```bash
POST /api/chat
Content-Type: application/json

{
  "data": [["سلام خوبی؟"]]
}
```

## 🛠️ اسکریپت‌ها

### آموزش یکباره

```bash
python scripts/train_once.py
```

### آموزش تدریجی

```bash
python scripts/train_incremental.py
```

### چت تعاملی

```bash
python scripts/chat.py
```

## ⚙️ تنظیمات

تنظیمات در `app/core/config.py` قابل تغییر است یا از متغیرهای محیطی:

```bash
export BASE_MODEL="HooshvareLab/gpt2-fa"
export API_HOST="0.0.0.0"
export API_PORT=8000
export DEFAULT_TEMPERATURE=0.9
```

## 📚 مستندات

مستندات کامل در فولدر `docs/`:

- `README_API.md` - مستندات API
- `INTEGRATION_GUIDE.md` - راهنمای یکپارچه‌سازی
- `HOW_IT_WORKS.md` - نحوه کار سیستم
- `MODEL_OPTIONS.md` - گزینه‌های مدل

## 🧪 تست

```bash
python -m pytest tests/
```

یا:

```bash
python tests/test_api.py
```

## 📝 ساختار کد

- **app/api/**: Endpoints و models
- **app/core/**: تنظیمات و بارگذاری مدل
- **app/services/**: منطق کسب‌وکار
- **scripts/**: اسکریپت‌های کمکی
- **tests/**: تست‌ها

## 🔧 توسعه

برای توسعه:

1. Fork کنید
2. Branch ایجاد کنید (`git checkout -b feature/AmazingFeature`)
3. Commit کنید (`git commit -m 'Add some AmazingFeature'`)
4. Push کنید (`git push origin feature/AmazingFeature`)
5. Pull Request باز کنید

## 📄 لایسنس

این پروژه برای استفاده آزاد است.

## 🤝 مشارکت

مشارکت‌ها خوش‌آمد هستند! لطفاً ابتدا یک issue باز کنید.

## 📧 تماس

برای سوالات و پیشنهادات، لطفاً issue باز کنید.

