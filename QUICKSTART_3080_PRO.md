# 🚀 راهنمای سریع Fine-tuning حرفه‌ای برای RTX 3080 10GB

## 📋 پیش‌نیازها

1. **نصب PyTorch با CUDA:**
   ```bash
   # برای CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # یا برای CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **نصب وابستگی‌ها:**
   ```bash
   pip install -r requirements_3080.txt
   ```

## 🎯 آموزش مدل

```bash
python scripts/train_3080.py
```

**زمان تقریبی:** 1.5 تا 2.5 ساعت

**خروجی:** `/home/arisa/paradiseModel/models/final_model/llama3_8b_persian_paradise`

## 💬 استفاده از مدل (Inference)

```bash
python scripts/inference_gradio.py
```

سپس مرورگر را باز کنید و به `http://localhost:7860` بروید.

## ⚙️ تنظیمات بهینه

- **4-bit Quantization** با bitsandbytes
- **QLoRA** با r=64, alpha=16
- **Gradient Checkpointing** برای صرفه‌جویی در حافظه
- **FP16** برای سرعت بیشتر
- **Paged AdamW 8-bit** optimizer

## 📝 نکات مهم

1. مدل به صورت خودکار بین `Meta-Llama-3-8B-Instruct` و `Hermes-2-Pro-Mistral-7B` انتخاب می‌کند
2. System Prompt احساسی به صورت خودکار به همه نمونه‌ها اضافه می‌شود
3. مدل در صورت کمبود حافظه GPU، از CPU offloading استفاده می‌کند

## 🎨 ویژگی‌های Inference

- ✅ پشتیبانی از آپلود عکس
- ✅ چت با تاریخچه
- ✅ تنظیمات پیشرفته Generation
- ✅ رابط کاربری زیبا با فونت Vazir

