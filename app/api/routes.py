#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API routes
"""
import json
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from app.api.models import ChatRequest, ChatResponse, HealthResponse
from app.services.chat_service import ChatService
from app.core.model_loader import is_model_loaded

router = APIRouter()
chat_service = ChatService()


@router.get("/", response_model=HealthResponse)
async def root():
    """بررسی وضعیت API"""
    return {
        "status": "running",
        "model_loaded": is_model_loaded()
    }


@router.get("/health", response_model=HealthResponse)
async def health():
    """بررسی سلامت API"""
    return {
        "status": "healthy" if is_model_loaded() else "model_not_loaded",
        "model_loaded": is_model_loaded()
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    چت با مدل
    
    Args:
        request: درخواست چت شامل message و پارامترهای generation
    
    Returns:
        پاسخ مدل
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response = await chat_service.generate_response(
            message=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            no_repeat_ngram_size=request.no_repeat_ngram_size,
        )
        
        return ChatResponse(response=response, status="success")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@router.post("/chat/simple")
async def chat_simple(message: str):
    """
    چت ساده (فقط message)
    
    Args:
        message: متن کاربر
    
    Returns:
        پاسخ مدل
    """
    request = ChatRequest(message=message)
    return await chat(request)


@router.post("/api/chat")
async def chat_gradio(request: Request):
    """
    Endpoint سازگار با Gradio
    دریافت داده‌های Gradio و برگرداندن پاسخ مناسب
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # دریافت داده‌های raw
        body = await request.json()
        
        # استخراج message از داده‌های Gradio
        message = None
        
        if "data" in body and isinstance(body["data"], list) and len(body["data"]) > 0:
            if isinstance(body["data"][0], list) and len(body["data"][0]) > 0:
                message = body["data"][0][0]
            elif isinstance(body["data"][0], str):
                message = body["data"][0]
        elif "message" in body:
            message = body["message"]
        elif isinstance(body, list) and len(body) > 0:
            if isinstance(body[0], list) and len(body[0]) > 0:
                message = body[0][0]
            elif isinstance(body[0], str):
                message = body[0]
        
        if not message or not str(message).strip():
            raise HTTPException(status_code=400, detail="Message not found in request")
        
        # استفاده از chat service
        response = await chat_service.generate_response(message=message)
        
        # برگرداندن پاسخ به فرمت Gradio: [["response"]]
        return [[response]]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint برای چت سریع و real-time
    """
    await websocket.accept()
    
    if not is_model_loaded():
        await websocket.send_json({
            "error": "Model not loaded",
            "status": "error"
        })
        await websocket.close()
        return
    
    try:
        while True:
            # دریافت پیام از client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message = message_data.get("message", "")
                
                if not message or not message.strip():
                    await websocket.send_json({
                        "error": "Message cannot be empty",
                        "status": "error"
                    })
                    continue
                
                # پارامترهای اختیاری
                max_tokens = message_data.get("max_tokens", 200)
                temperature = message_data.get("temperature", 1.0)
                top_p = message_data.get("top_p", 0.92)
                top_k = message_data.get("top_k", 40)
                repetition_penalty = message_data.get("repetition_penalty", 1.5)
                no_repeat_ngram_size = message_data.get("no_repeat_ngram_size", 4)
                
                # تولید پاسخ
                response = await chat_service.generate_response(
                    message=message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
                
                # ارسال پاسخ
                await websocket.send_json({
                    "response": response,
                    "status": "success"
                })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "error": "Invalid JSON format",
                    "status": "error"
                })
            except Exception as e:
                await websocket.send_json({
                    "error": f"Error generating response: {str(e)}",
                    "status": "error"
                })
                
    except WebSocketDisconnect:
        # Client disconnected normally
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "error": f"Connection error: {str(e)}",
                "status": "error"
            })
            await websocket.close()
        except:
            pass

