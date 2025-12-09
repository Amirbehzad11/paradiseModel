#!/bin/bash
# اسکریپت برای اجرای API به صورت service

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$API_DIR"

echo "🚀 Starting Chatbot API Service..."

# بررسی وجود مدل
if [ ! -d "./final_model" ]; then
    echo "❌ Model not found! Please run train_once.py first."
    exit 1
fi

# بررسی وجود Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    exit 1
fi

# اجرای API
echo "📡 Starting API on http://0.0.0.0:8000"
echo "📚 API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 api.py --host 0.0.0.0 --port 8000

