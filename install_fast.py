#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نصب سریع با آینه‌های ایرانی
Fast installation with Iranian mirrors
"""

import subprocess
import sys

# آینه‌های ایرانی
MIRRORS = [
    "https://pypi.rasa.ir/simple",
    "https://pypi.douban.com/simple",
    "https://mirrors.aliyun.com/pypi/simple",
]

def run_command(cmd, check=True):
    """اجرای دستور"""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def install_with_mirror(package, mirror=None):
    """نصب با آینه"""
    if mirror:
        cmd = f"pip install {package} -i {mirror} --trusted-host {mirror.split('//')[1].split('/')[0]}"
    else:
        cmd = f"pip install {package}"
    return run_command(cmd)

print("=" * 50)
print("🚀 نصب سریع با آینه‌های ایرانی")
print("🚀 Fast installation with Iranian mirrors")
print("=" * 50)

# ارتقای pip
print("\n⬆️  Upgrading pip...")
run_command("pip install --upgrade pip setuptools wheel")

# نصب PyTorch (از آینه اصلی - سریع‌تر)
print("\n🔥 Installing PyTorch with CUDA...")
run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# انتخاب آینه
mirror = MIRRORS[0]  # آینه اول
print(f"\n📦 Using mirror: {mirror}")

# نصب وابستگی‌های اصلی
print("\n📚 Installing main dependencies...")
packages = [
    "transformers>=4.40.0",
    "accelerate>=0.27.0",
    "peft>=0.8.0",
    "bitsandbytes>=0.43.0",
    "datasets>=2.18.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "tqdm>=4.66.0",
    "huggingface-hub>=0.20.0",
    "tokenizers>=0.15.0",
    "safetensors>=0.4.0",
]

for pkg in packages:
    install_with_mirror(pkg, mirror)

print("\n" + "=" * 50)
print("✅ نصب کامل شد!")
print("✅ Installation completed!")
print("=" * 50)
print("\nحالا می‌توانید اجرا کنید:")
print("Now you can run:")
print("  python train_once.py")

