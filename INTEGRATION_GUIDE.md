# راهنمای استفاده از API در پروژه‌های دیگر
# Integration Guide - Using API in Other Projects

## 🎯 استفاده از API در پروژه‌های دیگر

API شما روی یک port مشخص (مثلاً 8000) اجرا می‌شود و می‌توانید از **هر پروژه دیگری** با HTTP request به آن دسترسی داشته باشید.

## 🚀 راه‌اندازی API

### 1. اجرای API به صورت Background Service

**Linux/Ubuntu:**

```bash
# اجرا در background
nohup python api.py --host 0.0.0.0 --port 8000 > api.log 2>&1 &

# یا با screen
screen -S chatbot-api
python api.py --host 0.0.0.0 --port 8000
# Ctrl+A سپس D برای detach
```

**Windows:**

```powershell
# اجرا در PowerShell background
Start-Process python -ArgumentList "api.py","--host","0.0.0.0","--port","8000" -WindowStyle Hidden
```

**یا با PM2 (Node.js process manager):**

```bash
npm install -g pm2
pm2 start api.py --name chatbot-api --interpreter python -- --host 0.0.0.0 --port 8000
pm2 save
pm2 startup  # برای اجرای خودکار بعد از restart
```

---

## 💻 استفاده در پروژه‌های مختلف

### 1. PHP

```php
<?php
function chatWithModel($message) {
    $url = 'http://localhost:8000/chat';
    
    $data = [
        'message' => $message,
        'max_tokens' => 300,
        'temperature' => 0.7
    ];
    
    $options = [
        'http' => [
            'header'  => "Content-type: application/json\r\n",
            'method'  => 'POST',
            'content' => json_encode($data)
        ]
    ];
    
    $context = stream_context_create($options);
    $result = file_get_contents($url, false, $context);
    
    if ($result === FALSE) {
        return ['error' => 'API connection failed'];
    }
    
    $response = json_decode($result, true);
    return $response['response'];
}

// استفاده
$response = chatWithModel("سلام خوبی؟");
echo $response;
?>
```

**یا با cURL:**

```php
<?php
function chatWithModel($message) {
    $ch = curl_init('http://localhost:8000/chat');
    
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode([
        'message' => $message
    ]));
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json'
    ]);
    
    $response = curl_exec($ch);
    curl_close($ch);
    
    $data = json_decode($response, true);
    return $data['response'] ?? 'Error';
}
?>
```

---

### 2. Node.js / Express

```javascript
const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

// Endpoint که از API استفاده می‌کند
app.post('/my-chat', async (req, res) => {
    try {
        const { message } = req.body;
        
        const response = await axios.post('http://localhost:8000/chat', {
            message: message,
            max_tokens: 300,
            temperature: 0.7
        });
        
        res.json({
            success: true,
            response: response.data.response
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
```

---

### 3. Python (Flask/Django)

**Flask:**

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
API_URL = "http://localhost:8000/chat"

@app.route('/my-chat', methods=['POST'])
def my_chat():
    user_message = request.json.get('message')
    
    try:
        response = requests.post(API_URL, json={
            'message': user_message,
            'max_tokens': 300
        })
        
        return jsonify({
            'success': True,
            'response': response.json()['response']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=3000)
```

**Django:**

```python
# views.py
import requests
from django.http import JsonResponse

def chat_view(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        
        try:
            response = requests.post(
                'http://localhost:8000/chat',
                json={'message': message}
            )
            return JsonResponse({
                'success': True,
                'response': response.json()['response']
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
```

---

### 4. JavaScript (Frontend)

```javascript
// در frontend شما
async function sendMessage(message) {
    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                max_tokens: 300
            })
        });
        
        const data = await response.json();
        return data.response;
    } catch (error) {
        console.error('Error:', error);
        return 'خطا در ارتباط با سرور';
    }
}

// استفاده
sendMessage('سلام خوبی؟').then(response => {
    console.log(response);
});
```

---

### 5. C# / .NET

```csharp
using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class ChatService
{
    private readonly HttpClient _httpClient;
    private readonly string _apiUrl = "http://localhost:8000/chat";
    
    public ChatService()
    {
        _httpClient = new HttpClient();
    }
    
    public async Task<string> SendMessage(string message)
    {
        var requestData = new
        {
            message = message,
            max_tokens = 300,
            temperature = 0.7
        };
        
        var json = JsonConvert.SerializeObject(requestData);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        
        var response = await _httpClient.PostAsync(_apiUrl, content);
        var responseContent = await response.Content.ReadAsStringAsync();
        
        var result = JsonConvert.DeserializeObject<dynamic>(responseContent);
        return result.response;
    }
}
```

---

### 6. Java / Spring Boot

```java
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;

@Service
public class ChatService {
    private final String API_URL = "http://localhost:8000/chat";
    private final RestTemplate restTemplate = new RestTemplate();
    
    public String sendMessage(String message) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        Map<String, Object> request = new HashMap<>();
        request.put("message", message);
        request.put("max_tokens", 300);
        
        HttpEntity<Map<String, Object>> entity = 
            new HttpEntity<>(request, headers);
        
        Map<String, Object> response = restTemplate.postForObject(
            API_URL, entity, Map.class
        );
        
        return (String) response.get("response");
    }
}
```

---

## 🔧 تنظیمات CORS

اگر از frontend استفاده می‌کنید، باید CORS را تنظیم کنید:

**در `api.py`:**

```python
# برای production، origins را محدود کنید
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React app
        "http://localhost:8080",  # Vue app
        "https://yourdomain.com"  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

## 🌐 استفاده از Domain/Subdomain

اگر می‌خواهید API را روی یک domain خاص اجرا کنید:

### با Nginx (Reverse Proxy):

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

سپس در پروژه‌های دیگر:

```javascript
const API_URL = 'http://api.yourdomain.com';
// یا
const API_URL = 'https://api.yourdomain.com';  // با SSL
```

---

## 🔒 امنیت

### 1. API Key (اختیاری)

می‌توانید API Key اضافه کنید:

```python
# در api.py
API_KEY = os.getenv("API_KEY", "your-secret-key")

@app.post("/chat")
async def chat(request: ChatRequest, api_key: str = Header(None)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ... rest of code
```

**استفاده:**

```javascript
fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'api-key': 'your-secret-key'
    },
    body: JSON.stringify({message: 'سلام'})
})
```

### 2. Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("10/minute")  # 10 درخواست در دقیقه
async def chat(request: ChatRequest):
    # ... code
```

---

## 📊 Monitoring

برای monitoring API:

```bash
# با PM2
pm2 monit

# یا log ها
tail -f api.log
```

---

## ✅ چک‌لیست

- [ ] API روی port مشخص اجرا می‌شود
- [ ] CORS برای frontend تنظیم شده
- [ ] API در background اجرا می‌شود (nohup/pm2)
- [ ] Rate limiting اضافه شده (اختیاری)
- [ ] API Key برای امنیت (اختیاری)
- [ ] Monitoring و logging فعال است

---

**نکته**: API شما روی `http://localhost:8000` (یا IP سرور) در دسترس است و می‌توانید از **هر زبان برنامه‌نویسی** به آن دسترسی داشته باشید!

