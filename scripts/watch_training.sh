#!/bin/bash
# تماشای log به صورت زنده
# Watch training log in real-time

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/logs/training.pid"
STATUS_FILE="$PROJECT_DIR/logs/training_status.txt"

# پیدا کردن فایل log
if [ -f "$STATUS_FILE" ]; then
    LOG_FILE=$(grep "Log:" "$STATUS_FILE" 2>/dev/null | cut -d' ' -f2-)
fi

if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
    # پیدا کردن آخرین فایل log
    LOG_FILE=$(ls -t "$PROJECT_DIR/logs"/training_*.log 2>/dev/null | head -1)
fi

if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
    echo "❌ No log file found"
    exit 1
fi

echo "=================================================================================="
echo "👀 تماشای Log به صورت زنده (Watching Log)"
echo "=================================================================================="
echo "📝 Log file: $LOG_FILE"
echo "💡 Press Ctrl+C to stop"
echo "=================================================================================="
echo ""

tail -f "$LOG_FILE"

