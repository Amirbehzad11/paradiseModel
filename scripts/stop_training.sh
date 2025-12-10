#!/bin/bash
# متوقف کردن آموزش
# Stop training

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/logs/training.pid"
STATUS_FILE="$PROJECT_DIR/logs/training_status.txt"

echo "=================================================================================="
echo "🛑 متوقف کردن آموزش (Stop Training)"
echo "=================================================================================="
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "❌ Training is not running"
    echo "💡 No PID file found: $PID_FILE"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "⚠️  Process not found (PID: $PID)"
    echo "💡 Training may have already finished"
    rm -f "$PID_FILE"
    exit 0
fi

echo "📊 PID: $PID"
echo ""

# سوال از کاربر
read -p "⚠️  Are you sure you want to stop training? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 0
fi

# متوقف کردن
echo "🛑 Stopping training..."
kill "$PID"

# صبر برای توقف
sleep 2

# بررسی
if ps -p "$PID" > /dev/null 2>&1; then
    echo "⚠️  Process still running, force killing..."
    kill -9 "$PID"
    sleep 1
fi

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "✅ Training stopped successfully"
    rm -f "$PID_FILE"
    if [ -f "$STATUS_FILE" ]; then
        sed -i 's/Status: Running/Status: Stopped/' "$STATUS_FILE"
        echo "Stopped: $(date)" >> "$STATUS_FILE"
    fi
else
    echo "❌ Failed to stop training"
    exit 1
fi

