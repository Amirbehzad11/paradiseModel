# API Documentation
# مستندات API

## 🚀 راه‌اندازی
## Setup

### 1. نصب وابستگی‌ها

```bash
pip install fastapi uvicorn pydantic
```

یا:

```bash
pip install -r requirements.txt
```

### 2. اجرای API

```bash
python api.py
```

یا با تنظیمات خاص:

```bash
python api.py --host 0.0.0.0 --port 8000
```

API روی `http://localhost:8000` اجرا می‌شود.

### 3. مشاهده مستندات

بعد از اجرا، مستندات خودکار API در دسترس است:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📡 Endpoints

### 1. Health Check

**GET** `/health`

بررسی وضعیت API و مدل

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**مثال:**
```bash
curl http://localhost:8000/health
```

---

### 2. Chat (پیشنهادی)

**POST** `/chat`

چت با مدل با کنترل کامل پارامترها

**Request Body:**
```json
{
  "message": "سلام خوبی؟",
  "max_tokens": 300,
  "temperature": 0.7,
  "top_p": 0.9,
  "repetition_penalty": 1.2
}
```

**Response:**
```json
{
  "response": "سلام عزیزم… من اینجام. همیشه پیشتم...",
  "status": "success"
}
```

**مثال:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "سلام خوبی؟",
    "max_tokens": 300
  }'
```

**پارامترها:**
- `message` (required): متن کاربر
- `max_tokens` (optional, default: 300): حداکثر تعداد token در پاسخ
- `temperature` (optional, default: 0.7): کنترل خلاقیت (0.1-1.0)
- `top_p` (optional, default: 0.9): nucleus sampling
- `repetition_penalty` (optional, default: 1.2): جلوگیری از تکرار

---

### 3. Chat Simple

**POST** `/chat/simple?message=سلام`

چت ساده (فقط message)

**مثال:**
```bash
curl -X POST "http://localhost:8000/chat/simple?message=سلام خوبی؟"
```

---

## 💻 مثال‌های استفاده
## Usage Examples

### Python

```python
import requests

# چت ساده
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "سلام خوبی؟",
        "max_tokens": 300
    }
)

result = response.json()
print(result["response"])
```

### JavaScript (Fetch)

```javascript
fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'سلام خوبی؟',
    max_tokens: 300
  })
})
.then(response => response.json())
.then(data => console.log(data.response));
```

### JavaScript (Axios)

```javascript
const axios = require('axios');

axios.post('http://localhost:8000/chat', {
  message: 'سلام خوبی؟',
  max_tokens: 300
})
.then(response => {
  console.log(response.data.response);
});
```

### cURL

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "سلام خوبی؟"}'
```

---

## 🔧 تنظیمات
## Configuration

### تغییر Host و Port

```bash
python api.py --host 0.0.0.0 --port 8080
```

### Auto-reload (برای توسعه)

```bash
python api.py --reload
```

---

## 🛡️ امنیت
## Security

### CORS

در حال حاضر CORS برای همه origins فعال است. برای production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # محدود کردن
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

### Rate Limiting

برای جلوگیری از سوء استفاده، می‌توانید rate limiting اضافه کنید:

```bash
pip install slowapi
```

---

## 🐛 عیب‌یابی
## Troubleshooting

### خطای "Model not loaded"

مطمئن شوید:
1. `./final_model` وجود دارد
2. مدل با `train_once.py` train شده است

### خطای Port در حال استفاده

```bash
# تغییر port
python api.py --port 8001
```

### تست API

```bash
python test_api.py
```

---

## 📊 Performance

- **اولین درخواست**: ممکن است کند باشد (warmup)
- **درخواست‌های بعدی**: سریع (~1-3 ثانیه)
- **Concurrent requests**: پشتیبانی می‌شود

---

## 🔄 استفاده در Production

### با Gunicorn

```bash
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### با Docker

```dockerfile
FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "api.py", "--host", "0.0.0.0", "--port", "8000"]
```

---

**نکته**: برای استفاده در production، حتماً امنیت و rate limiting را اضافه کنید!

