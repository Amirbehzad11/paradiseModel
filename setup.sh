#!/bin/bash

# اسکریپت نصب کامل سیستم - اجرای یک دستوری
# Complete system installation script - one command execution

set -e  # در صورت خطا متوقف شود
# Stop on error

echo "=========================================="
echo "🚀 شروع نصب سیستم آموزش و چت بات معنوی"
echo "🚀 Starting Spiritual Chatbot System Installation"
echo "=========================================="

# بررسی وجود Python 3.8+
# Check for Python 3.8+
if ! command -v python3 &> /dev/null; then
    echo "❌ خطا: Python 3 یافت نشد. لطفا Python 3.8+ نصب کنید."
    echo "❌ Error: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $PYTHON_VERSION"

# بررسی وجود NVIDIA GPU
# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
fi

# ایجاد محیط مجازی Python
# Create Python virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# فعال‌سازی محیط مجازی
# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# ارتقای pip
# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# نصب PyTorch با پشتیبانی CUDA
# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# نصب وابستگی‌ها
# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# بررسی نصب BitsAndBytes
# Verify BitsAndBytes installation
echo "🔍 Verifying BitsAndBytes installation..."
python3 -c "import bitsandbytes; print('✅ BitsAndBytes installed successfully')" || echo "⚠️  BitsAndBytes verification failed"

# ایجاد دایرکتوری‌های لازم
# Create necessary directories
echo "📁 Creating directories..."
mkdir -p final_model
mkdir -p checkpoints

# تنظیم متغیرهای محیطی
# Set environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not set. You may need to login to Hugging Face."
    echo "   Run: huggingface-cli login"
    echo "   Or set: export HF_TOKEN=your_token_here"
fi

echo ""
echo "=========================================="
echo "✅ نصب با موفقیت انجام شد!"
echo "✅ Installation completed successfully!"
echo "=========================================="
echo ""
echo "📝 مراحل بعدی:"
echo "📝 Next steps:"
echo ""
echo "1. فعال‌سازی محیط مجازی:"
echo "   source venv/bin/activate"
echo ""
echo "2. آموزش مدل (فقط یک بار):"
echo "   python train_once.py"
echo ""
echo "3. شروع چت:"
echo "   python chat.py"
echo ""
echo "=========================================="

