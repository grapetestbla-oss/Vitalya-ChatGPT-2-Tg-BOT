#!/usr/bin/env python3
"""
Скрипт для запуска нейросетевого Telegram бота "Виталя"
"""

import os
import sys
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def check_dependencies():
    """Проверка наличия необходимых зависимостей"""
    missing_deps = []
    
    try:
        import telegram
    except ImportError:
        missing_deps.append('python-telegram-bot')
    
    try:
        import transformers
    except ImportError:
        missing_deps.append('transformers')
    
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    if missing_deps:
        print(f"Отсутствуют зависимости: {', '.join(missing_deps)}")
        print("Установите их с помощью команды:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Основная функция запуска бота"""
    print("Загрузка нейросетевого Telegram бота Виталя...")
    
    # Проверяем зависимости
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Импортируем и запускаем основной модуль бота
        from tg_neuro_bot import main as bot_main
        print("Бот успешно загружен. Запуск...")
        bot_main()
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Убедитесь, что все зависимости установлены:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка запуска бота: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()