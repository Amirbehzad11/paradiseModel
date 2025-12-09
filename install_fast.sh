#!/bin/bash

# اسکریپت نصب سریع با آینه‌های ایرانی
# Fast installation script with Iranian mirrors

set -e

echo "=========================================="
echo "🚀 نصب سریع با آینه‌های ایرانی"
echo "🚀 Fast installation with Iranian mirrors"
echo "=========================================="

# تنظیم آینه‌های pip ایرانی
# Set Iranian pip mirrors
export PIP_INDEX_URL="https://pypi.rasa.ir/simple"
export PIP_TRUSTED_HOST="pypi.rasa.ir"

# یا از آینه‌های دیگر:
# export PIP_INDEX_URL="https://pypi.douban.com/simple"
# export PIP_TRUSTED_HOST="pypi.douban.com"

echo "📦 استفاده از آینه: $PIP_INDEX_URL"
echo "📦 Using mirror: $PIP_INDEX_URL"

# ارتقای pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST

# نصب PyTorch با CUDA (از آینه اصلی - سریع‌تر)
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# نصب وابستگی‌های اصلی به صورت موازی
echo "📚 Installing main dependencies..."
pip install transformers>=4.40.0 -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST
pip install accelerate>=0.27.0 -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST
pip install peft>=0.8.0 -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST
pip install bitsandbytes>=0.43.0 -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST
pip install datasets>=2.18.0 -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST

# نصب وابستگی‌های دیگر
echo "📚 Installing additional dependencies..."
pip install sentencepiece>=0.1.99 protobuf>=3.20.0 scipy>=1.11.0 scikit-learn>=1.3.0 numpy>=1.24.0 pandas>=2.0.0 tqdm>=4.66.0 huggingface-hub>=0.20.0 tokenizers>=0.15.0 safetensors>=0.4.0 -i $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST

echo ""
echo "✅ نصب کامل شد!"
echo "✅ Installation completed!"
echo ""
echo "حالا می‌توانید اجرا کنید:"
echo "Now you can run:"
echo "  python train_once.py"

