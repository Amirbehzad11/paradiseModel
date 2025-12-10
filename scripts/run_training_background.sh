#!/bin/bash
# اجرای آموزش در background با امکان چک کردن وضعیت
# Run training in background with status checking

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$SCRIPT_DIR/train_3080.py"
LOG_FILE="$PROJECT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$PROJECT_DIR/logs/training.pid"
STATUS_FILE="$PROJECT_DIR/logs/training_status.txt"

# ایجاد پوشه logs
mkdir -p "$PROJECT_DIR/logs"

# بررسی اینکه آیا قبلاً در حال اجرا است
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Training is already running (PID: $OLD_PID)"
        echo "💡 Use './scripts/check_training.sh' to check status"
        echo "💡 Use './scripts/stop_training.sh' to stop it"
        exit 1
    else
        # PID قدیمی وجود ندارد، فایل را پاک کن
        rm -f "$PID_FILE"
    fi
fi

# شروع آموزش در background
echo "🚀 Starting training in background..."
echo "📝 Log file: $LOG_FILE"
echo ""

cd "$PROJECT_DIR"

# اجرا در background و ذخیره PID
nohup python3 "$TRAIN_SCRIPT" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

# ذخیره PID
echo $TRAIN_PID > "$PID_FILE"

# ذخیره اطلاعات
echo "Started: $(date)" > "$STATUS_FILE"
echo "PID: $TRAIN_PID" >> "$STATUS_FILE"
echo "Log: $LOG_FILE" >> "$STATUS_FILE"
echo "Status: Running" >> "$STATUS_FILE"

echo "✅ Training started!"
echo "📊 PID: $TRAIN_PID"
echo "📝 Log: $LOG_FILE"
echo ""
echo "💡 Commands:"
echo "   Check status:  tail -f $LOG_FILE"
echo "   Check status:  ./scripts/check_training.sh"
echo "   Stop training: ./scripts/stop_training.sh"

