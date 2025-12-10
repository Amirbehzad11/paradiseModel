#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اجرای خودکار سیستم یادگیری
Auto-run learning system (برای استفاده در cron یا scheduled tasks)
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.smart_learning_system import main

if __name__ == "__main__":
    main()

