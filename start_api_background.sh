#!/bin/bash
# اجرای API در background

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$API_DIR"

echo "🚀 Starting Chatbot API in background..."

# بررسی وجود مدل
if [ ! -d "./final_model" ]; then
    echo "❌ Model not found! Please run train_once.py first."
    exit 1
fi

# اجرا در background
nohup python3 api.py --host 0.0.0.0 --port 8000 > api.log 2>&1 &

PID=$!
echo "✅ API started with PID: $PID"
echo "📝 Logs: tail -f api.log"
echo "🛑 Stop: kill $PID"
echo ""
echo "API running on: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"

# ذخیره PID
echo $PID > api.pid

