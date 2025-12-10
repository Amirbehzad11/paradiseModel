#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نقطه ورود ساده برای اجرای API
Simple entry point for running the API
"""
from app.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

