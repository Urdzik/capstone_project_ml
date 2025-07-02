#!/usr/bin/env python3
"""
Швидкий запуск тренування великих моделей
"""

import sys
import os

# Додаємо src до path
sys.path.append('src')

from src.train_large_model import main

if __name__ == "__main__":
    print("🚀 Запуск тренування великих ML моделей...")
    print("📊 Це може зайняти 10-30 хвилин залежно від швидкості інтернету")
    print("💾 Моделі будуть збережені в data/models/")
    print()
    
    main() 