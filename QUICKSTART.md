# راهنمای سریع - Quick Start Guide

## 🚀 اجرای سریع API

### 1. نصب وابستگی‌ها

```bash
pip install -r requirements.txt
```

### 2. آموزش مدل (اگر مدل وجود ندارد)

```bash
python scripts/train_once.py
```

### 3. اجرای API

```bash
python run.py
```

یا:

```bash
python -m app.main
```

### 4. دسترسی به API

- API: http://localhost:8000
- مستندات: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 📝 مثال استفاده

### با curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "سلام خوبی؟"}'
```

### با Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "سلام خوبی؟"}
)

print(response.json()["response"])
```

## 🛠️ اسکریپت‌های مفید

- `python scripts/chat.py` - چت تعاملی
- `python scripts/train_once.py` - آموزش یکباره
- `python tests/test_api.py` - تست API

## 📚 مستندات کامل

برای اطلاعات بیشتر، به فولدر `docs/` مراجعه کنید.

