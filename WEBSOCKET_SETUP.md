# راهنمای WebSocket API

## 🚀 دستورات سریع

### 1. اجرای API

```bash
cd C:\xampp\htdocs\koshaHosh\TTS_MODEL
python run.py
```

API روی `http://localhost:8000` اجرا می‌شود.

### 2. Endpoint های موجود

- **HTTP**: `POST http://localhost:8000/chat`
- **WebSocket**: `ws://localhost:8000/ws/chat` ⚡ (سریع‌تر)

### 3. تست WebSocket

```python
import websocket
import json

ws = websocket.create_connection("ws://localhost:8000/ws/chat")
ws.send(json.dumps({"message": "سلام"}))
response = ws.recv()
print(json.loads(response))
ws.close()
```

---

## 📝 فرمت پیام WebSocket

**ارسال:**
```json
{
  "message": "متن پیام شما",
  "max_tokens": 300,
  "temperature": 1.0,
  "top_p": 0.92,
  "top_k": 40,
  "repetition_penalty": 1.5,
  "no_repeat_ngram_size": 4
}
```

**دریافت:**
```json
{
  "response": "پاسخ مدل",
  "status": "success"
}
```

یا در صورت خطا:
```json
{
  "error": "پیام خطا",
  "status": "error"
}
```

---

## ✅ آماده برای استفاده با MuseTalk

API شما اکنون از WebSocket پشتیبانی می‌کند و MuseTalk به صورت خودکار از آن استفاده می‌کند!

