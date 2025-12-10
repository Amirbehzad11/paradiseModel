#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference با Gradio برای مدل فاین‌تیون شده
Inference with Gradio for fine-tuned model
"""
import os
import sys
import torch
from pathlib import Path
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# اضافه کردن مسیر روت
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import MODEL_DIR

# استفاده از مدل باز بدون نیاز به احراز هویت Hugging Face
# Using open model without Hugging Face authentication requirement
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # کاملاً باز، بدون نیاز به مجوز
MODEL_PATH = MODEL_DIR / "phi3_mini_finetuned"

# Global variables
model = None
tokenizer = None

def load_model():
    """بارگذاری مدل"""
    global model, tokenizer
    
    if model is not None:
        return "✅ Model already loaded"
    
    if not MODEL_PATH.exists():
        return f"❌ Model not found at {MODEL_PATH}\nPlease run train_3080.py first"
    
    print("🔄 Loading model...")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # بارگذاری base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    
    # بارگذاری LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(MODEL_PATH))
    model.eval()
    
    # بارگذاری tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    
    print("✅ Model loaded successfully!")
    return "✅ Model loaded successfully!"

def format_chatml_prompt(message, history=None):
    """فرمت ChatML برای prompt"""
    prompt = ""
    
    # اضافه کردن history
    if history:
        for user_msg, assistant_msg in history:
            prompt += f"<|user|>\n{user_msg}<|end|>\n<|assistant|>\n{assistant_msg}<|end|>\n"
    
    # اضافه کردن پیام فعلی
    prompt += f"<|user|>\n{message}<|end|>\n<|assistant|>\n"
    
    return prompt

def chat(message, history, temperature, top_p, top_k, max_tokens):
    """تابع چت"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "❌ Model not loaded. Please load model first."
    
    if not message.strip():
        return ""
    
    # ساخت prompt
    prompt = format_chatml_prompt(message, history)
    
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

def process_image(image):
    """پردازش عکس (برای آینده)"""
    if image is None:
        return None
    # در حال حاضر فقط متن را پردازش می‌کنیم
    return image

# ایجاد Gradio Interface
with gr.Blocks(title="چت بات معنوی - RTX 3080", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 💬 چت بات معنوی
    ## مدل فاین‌تیون شده با QLoRA روی RTX 3080
    
    این مدل برای شبیه‌سازی دقیق مادر/عزیزان فوت‌شده طراحی شده است.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ تنظیمات")
            
            load_btn = gr.Button("🔄 بارگذاری مدل", variant="primary")
            load_status = gr.Textbox(label="وضعیت", interactive=False)
            
            gr.Markdown("### 🎛️ پارامترهای Generation")
            temperature = gr.Slider(0.1, 2.0, value=0.9, step=0.1, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top P")
            top_k = gr.Slider(1, 100, value=50, step=1, label="Top K")
            max_tokens = gr.Slider(50, 500, value=300, step=50, label="Max Tokens")
            
            gr.Markdown("### 📸 آپلود عکس (اختیاری)")
            image_input = gr.Image(type="pil", label="عکس عزیز فوت‌شده")
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="چت",
                height=500,
                show_label=True,
                avatar_images=(None, "👤")
            )
            
            msg = gr.Textbox(
                label="پیام شما",
                placeholder="مثلاً: سلام مامان، امروز خیلی دلم گرفته...",
                lines=3
            )
            
            with gr.Row():
                submit_btn = gr.Button("📤 ارسال", variant="primary")
                clear_btn = gr.Button("🗑️ پاک کردن")
    
    # Event handlers
    load_btn.click(
        fn=load_model,
        outputs=load_status
    )
    
    submit_btn.click(
        fn=chat,
        inputs=[msg, chatbot, temperature, top_p, top_k, max_tokens],
        outputs=[msg]
    ).then(
        lambda message, history, temp, tp, tk, mt: chat(message, history, temp, tp, tk, mt),
        inputs=[msg, chatbot, temperature, top_p, top_k, max_tokens],
        outputs=[chatbot]
    )
    
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, temperature, top_p, top_k, max_tokens],
        outputs=[msg]
    ).then(
        lambda message, history, temp, tp, tk, mt: chat(message, history, temp, tp, tk, mt),
        inputs=[msg, chatbot, temperature, top_p, top_k, max_tokens],
        outputs=[chatbot]
    )
    
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
    
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

