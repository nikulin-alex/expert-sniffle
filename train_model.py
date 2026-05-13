#!/usr/bin/env python3
"""
Скрипт обучения ML-модели для предсказания стратегий снижения цены.

Использование:
    python train_model.py <путь_к_данным.json> [путь_к_сохранению_модели.json]

Аргументы:
    путь_к_данным.json      - JSON файл с историческими данными закупок
    путь_к_сохранению...   - (опционально) Путь для сохранения модели.
                             По умолчанию: procurement_ai/model.json
"""

import sys
import json
import os

# Добавляем корень проекта в path для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from procurement_ai.ml_model import ReductionStrategyPredictor
from procurement_ai.utils import parse_price, extract_final_price


def calculate_reduction_percent(record: dict) -> float:
    """
    Вычисляет процент снижения цены на основе НМЦК и итоговой цены контракта.
    """
    nmcc_str = record.get('Начальная (максимальная) цена контракта', '0')
    nmcc = parse_price(nmcc_str)
    
    if nmcc <= 0:
        return 0.0
    
    final_price = extract_final_price(record)
    
    if final_price <= 0:
        return 0.0
    
    # Процент снижения = ((НМЦК - Итоговая) / НМЦК) * 100
    reduction = ((nmcc - final_price) / nmcc) * 100
    
    # Ограничиваем разумными пределами
    return max(0.0, min(100.0, reduction))


def load_data(file_path: str) -> list:
    """Загружает данные из JSON файла."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Если данные приходят в виде словаря с ключом (например, {'data': [...]}),
    # извлекаем список. Иначе считаем, что загружен сразу список.
    if isinstance(data, dict):
        # Пытаемся найти список внутри словаря
        for key in ['data', 'items', 'records', 'procurements']:
            if key in data and isinstance(data[key], list):
                return data[key]
        # Если ключи не найдены, пробуем вернуть первое значение, если это список
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, list):
            return first_val
        raise ValueError("Не удалось извлечь список закупок из JSON файла.")
    
    if isinstance(data, list):
        return data
    
    raise ValueError("JSON файл должен содержать список закупок или словарь со списком.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Ошибка: Не указан путь к файлу с данными.")
        sys.exit(1)

    data_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "procurement_ai/model.json"

    print(f"--- Запуск обучения модели ---")
    print(f"Источник данных: {data_file}")
    print(f"Путь сохранения модели: {model_path}")

    try:
        # 1. Загрузка данных
        print("\n[1/4] Загрузка данных...")
        procurements = load_data(data_file)
        print(f"Загружено записей: {len(procurements)}")

        if len(procurements) < 5:
            print("Предупреждение: Слишком мало данных для качественного обучения (< 5).")
        
        # 2. Инициализация и обучение
        print("\n[2/4] Подготовка данных (расчет процента снижения)...")
        
        # Добавляем поле reduction_percent к каждой записи
        valid_records = []
        skipped_count = 0
        for record in procurements:
            reduction = calculate_reduction_percent(record)
            if reduction > 0:  # Только записи с положительным снижением
                record['reduction_percent'] = reduction
                valid_records.append(record)
            else:
                skipped_count += 1
        
        print(f"Записей с рассчитанным снижением: {len(valid_records)}")
        if skipped_count > 0:
            print(f"Пропущено записей (нет данных о снижении): {skipped_count}")
        
        if len(valid_records) < 5:
            print("Предупреждение: Слишком мало данных для качественного обучения (< 5).")
            sys.exit(1)
        
        print("\n[3/4] Инициализация модели...")
        predictor = ReductionStrategyPredictor()

        print("\n[4/4] Обучение модели (градиентный спуск)...")
        # verbose=True выведет информацию о процессе обучения
        history = predictor.train(valid_records, verbose=True)

        # 4. Сохранение модели
        print(f"\n[5/5] Сохранение модели в {model_path}...")
        # Создаем директорию, если она не существует
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        predictor.save_model(model_path)
        print("Модель успешно сохранена!")

        # 4. Краткий отчет
        print("\n--- Отчет об обучении ---")
        print(f"Количество эпох: {len(history.get('loss_history', []))}")
        if history.get('loss_history'):
            final_loss = history['loss_history'][-1]
            print(f"Финальная ошибка (MSE): {final_loss:.6f}")
        
        print("\nГотово! Теперь вы можете использовать модель для предсказаний.")
        print(f"Пример использования в коде:")
        print(f"   from procurement_ai import ReductionStrategyPredictor")
        print(f"   predictor = ReductionStrategyPredictor()")
        print(f"   predictor.load_model('{model_path}')")
        print(f"   result = predictor.predict(new_record)")

    except FileNotFoundError as e:
        print(f"\nОшибка доступа к файлу: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nОшибка формата данных: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nКритическая ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
