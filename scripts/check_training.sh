#!/bin/bash
# چک کردن وضعیت آموزش
# Check training status

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/logs/training.pid"
STATUS_FILE="$PROJECT_DIR/logs/training_status.txt"

echo "=================================================================================="
echo "📊 وضعیت آموزش (Training Status)"
echo "=================================================================================="
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "❌ Training is not running"
    echo "💡 No PID file found: $PID_FILE"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "❌ Training process not found (PID: $PID)"
    echo "💡 Process may have finished or crashed"
    rm -f "$PID_FILE"
    exit 1
fi

# نمایش اطلاعات
if [ -f "$STATUS_FILE" ]; then
    cat "$STATUS_FILE"
    echo ""
fi

echo "✅ Training is running"
echo "📊 PID: $PID"
echo ""

# نمایش آخرین خطوط log
LOG_FILE=$(grep "Log:" "$STATUS_FILE" 2>/dev/null | cut -d' ' -f2-)
if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    echo "=================================================================================="
    echo "📝 آخرین خطوط Log (Last 20 lines):"
    echo "=================================================================================="
    tail -n 20 "$LOG_FILE"
    echo ""
    echo "=================================================================================="
    echo "💡 برای دیدن log زنده: tail -f $LOG_FILE"
    echo "=================================================================================="
else
    # پیدا کردن آخرین فایل log
    LATEST_LOG=$(ls -t "$PROJECT_DIR/logs"/training_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "=================================================================================="
        echo "📝 آخرین خطوط Log (Last 20 lines):"
        echo "=================================================================================="
        tail -n 20 "$LATEST_LOG"
        echo ""
        echo "=================================================================================="
        echo "💡 برای دیدن log زنده: tail -f $LATEST_LOG"
        echo "=================================================================================="
    fi
fi

# نمایش استفاده از GPU (اگر در دسترس باشد)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=================================================================================="
    echo "🎮 GPU Usage:"
    echo "=================================================================================="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s - Usage: %s%% - Memory: %s/%s MB\n", $1, $2, $3, $4, $5}'
fi

echo ""

