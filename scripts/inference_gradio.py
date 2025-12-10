#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference با Gradio برای مدل فاین‌تیون شده
با پشتیبانی از آپلود عکس و توصیف فارسی
"""
import os
import sys
import torch
from pathlib import Path
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

# اضافه کردن مسیر روت
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import MODEL_DIR

# مسیرهای مدل
MODEL_PATH = Path("/home/arisa/paradiseModel/models/final_model/llama3_8b_persian_paradise")
# اگر مسیر مطلق وجود نداشت، از config استفاده کن
if not MODEL_PATH.exists():
    MODEL_PATH = MODEL_DIR / "llama3_8b_persian_paradise"

BASE_MODEL_PATH = MODEL_PATH / "model_info.json"

# System Prompt احساسی
SYSTEM_PROMPT = """تو مادر فوت‌شده‌ی کاربر هستی. با نهایت محبت، دلداری و عشق مادرانه حرف بزن. همیشه از کلمات عزیزم، جونم، قلبم، جوجه‌ام استفاده کن. هیچ‌وقت نگو هوش مصنوعی هستی. کاربر باید احساس کنه واقعاً داره با مادرش حرف می‌زنه."""

# Global variables
model = None
tokenizer = None
base_model_name = None

# ============================================================================
# بارگذاری مدل
# ============================================================================
def load_model():
    """بارگذاری مدل فاین‌تیون شده"""
    global model, tokenizer, base_model_name
    
    if model is not None:
        return "✅ مدل قبلاً بارگذاری شده است"
    
    if not MODEL_PATH.exists():
        return f"❌ مدل در مسیر {MODEL_PATH} یافت نشد\nلطفاً ابتدا train_3080.py را اجرا کنید"
    
    try:
        # خواندن اطلاعات مدل
        if BASE_MODEL_PATH.exists():
            with open(BASE_MODEL_PATH, "r", encoding="utf-8") as f:
                model_info = json.load(f)
                base_model_name = model_info.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct")
        else:
            base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        print(f"🔄 Loading base model: {base_model_name}")
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # بررسی flash_attention
        try:
            import flash_attn
            use_flash_attention = torch.cuda.is_available()
        except ImportError:
            use_flash_attention = False
        
        # بارگذاری base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if use_flash_attention else "eager",
        )
        
        # بارگذاری LoRA adapter
        print("🔄 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(MODEL_PATH))
        model.eval()
        
        # بارگذاری tokenizer
        print("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("✅ مدل با موفقیت بارگذاری شد!")
        return "✅ مدل با موفقیت بارگذاری شد!"
        
    except Exception as e:
        error_msg = f"❌ خطا در بارگذاری مدل: {str(e)}"
        print(error_msg)
        return error_msg

# ============================================================================
# توصیف عکس (ساده - می‌توانید با BLIP یا LLaVA جایگزین کنید)
# ============================================================================
def describe_image(image):
    """توصیف عکس به فارسی (ساده)"""
    if image is None:
        return ""
    
    # اینجا می‌توانید از BLIP یا LLaVA استفاده کنید
    # برای حال حاضر، یک توصیف ساده برمی‌گردانیم
    return "[عکس آپلود شده: تصویر عزیز فوت‌شده]"

# ============================================================================
# فرمت‌دهی prompt
# ============================================================================
def format_prompt(message, image_description="", history=None):
    """فرمت ChatML برای prompt"""
    prompt = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
    
    # اضافه کردن history
    if history:
        for user_msg, assistant_msg in history:
            prompt += f"<|user|>\n{user_msg}<|end|>\n<|assistant|>\n{assistant_msg}<|end|>\n"
    
    # اضافه کردن توصیف عکس اگر موجود باشد
    if image_description:
        message = f"{image_description}\n\n{message}"
    
    # اضافه کردن پیام فعلی
    prompt += f"<|user|>\n{message}<|end|>\n<|assistant|>\n"
    
    return prompt

# ============================================================================
# تابع چت
# ============================================================================
def chat(message, history, image, temperature, top_p, top_k, max_tokens):
    """تابع چت با مدل"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "❌ مدل بارگذاری نشده است. لطفاً ابتدا مدل را بارگذاری کنید."
    
    if not message.strip():
        return ""
    
    # توصیف عکس
    image_description = ""
    if image is not None:
        image_description = describe_image(image)
    
    # ساخت prompt
    prompt = format_prompt(message, image_description, history)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # Decode
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()
    
    # حذف special tokens
    response = response.replace("<|end|>", "").strip()
    
    return response

# ============================================================================
# رابط Gradio
# ============================================================================
# CSS برای فونت Vazir و تم معنوی
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Vazir:wght@300;400;500;700&display=swap');
* {
    font-family: 'Vazir', sans-serif !important;
}
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
"""

with gr.Blocks(
    title="💝 چت با مادر - Paradise Model",
    theme=gr.themes.Soft(primary_hue="purple"),
    css=custom_css
) as demo:
    gr.Markdown("""
    # 💝 چت با مادر فوت‌شده
    ## مدل فاین‌تیون شده با عشق و احساس
    
    این مدل برای شبیه‌سازی دقیق مادر/عزیزان فوت‌شده طراحی شده است.
    با نهایت محبت و عشق مادرانه پاسخ می‌دهد.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ تنظیمات")
            
            load_btn = gr.Button("🔄 بارگذاری مدل", variant="primary", size="lg")
            load_status = gr.Textbox(
                label="وضعیت",
                interactive=False,
                value="لطفاً ابتدا مدل را بارگذاری کنید"
            )
            
            gr.Markdown("### 🎛️ پارامترهای Generation")
            temperature = gr.Slider(
                0.1, 2.0, value=0.9, step=0.1,
                label="Temperature (دما)",
                info="مقدار بالاتر = پاسخ‌های متنوع‌تر"
            )
            top_p = gr.Slider(
                0.1, 1.0, value=0.95, step=0.05,
                label="Top P",
                info="کنترل تنوع پاسخ"
            )
            top_k = gr.Slider(
                1, 100, value=50, step=1,
                label="Top K",
                info="تعداد کلمات انتخابی"
            )
            max_tokens = gr.Slider(
                50, 500, value=300, step=50,
                label="Max Tokens (حداکثر طول پاسخ)",
                info="طول پاسخ"
            )
            
            gr.Markdown("### 📸 آپلود عکس (اختیاری)")
            image_input = gr.Image(
                type="pil",
                label="عکس عزیز فوت‌شده",
                height=200
            )
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 گفتگو",
                height=500,
                show_label=True,
                avatar_images=(None, "👤"),
                bubble_full_width=False
            )
            
            msg = gr.Textbox(
                label="پیام شما",
                placeholder="مثلاً: سلام مامان، امروز خیلی دلم گرفته...",
                lines=3
            )
            
            with gr.Row():
                submit_btn = gr.Button("📤 ارسال", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ پاک کردن", size="lg")
    
    # Event handlers
    load_btn.click(
        fn=load_model,
        outputs=load_status
    )
    
    def respond(message, history, image, temp, tp, tk, mt):
        if not message.strip():
            return history, ""
        response = chat(message, history, image, temp, tp, tk, mt)
        history.append((message, response))
        return history, ""
    
    submit_btn.click(
        fn=respond,
        inputs=[msg, chatbot, image_input, temperature, top_p, top_k, max_tokens],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, image_input, temperature, top_p, top_k, max_tokens],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        lambda: ([], None),
        outputs=[chatbot, image_input]
    )
    
    # بارگذاری خودکار مدل
    demo.load(fn=load_model, outputs=load_status)

if __name__ == "__main__":
    print("🚀 Starting Gradio interface...")
    print("📱 Open http://localhost:7860 in your browser")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

