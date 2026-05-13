#!/usr/bin/env python3
"""
Скрипт для поиска схожих закупок и формирования аналитической сводки.

Использование:
    python find_similar.py path/to/input.json [--output result.json]

Входной файл (JSON) должен содержать запись о закупке в следующем формате:
{
    "reg_number": "...",
    "Организация, осуществляющая размещение": "...",
    "Регион": "...",
    "Наименование объекта закупки": "...",
    "Начальная (максимальная) цена контракта": "...",
    ...
}

Или можно передать массив исторических данных для загрузки в базу:
{
    "history": [...],
    "query": { ...запись для поиска... }
}
"""

import json
import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent))

from procurement_ai import ProcurementAnalyzer


def parse_price(price_str: str) -> float:
    """Парсинг строки цены в число."""
    if not price_str:
        return 0.0
    # Удаляем пробелы и валюту
    cleaned = price_str.replace(" ", "").replace("РУБЛЬ", "").replace("₽", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def extract_query_params(record: dict) -> dict:
    """Извлечение параметров поиска из записи."""
    params = {}
    
    # Заказчик
    customer_fields = [
        "Организация, осуществляющая размещение",
        "Заказчик",
        "customer"
    ]
    for field in customer_fields:
        if field in record and record[field]:
            params["customer"] = record[field]
            break
    
    # Регион
    region_fields = ["Регион", "region"]
    for field in region_fields:
        if field in record and record[field]:
            params["region"] = record[field]
            break
    
    # Вид работ / Наименование
    work_fields = [
        "Наименование объекта закупки",
        "Описание объекта закупки",
        "work_type",
        "object_name"
    ]
    for field in work_fields:
        if field in record and record[field]:
            params["work_type"] = record[field]
            break
    
    # НМЦК
    nmck_fields = [
        "Начальная (максимальная) цена контракта",
        "НМЦК",
        "nmck",
        "initial_price"
    ]
    nmck_value = 0.0
    for field in nmck_fields:
        if field in record and record[field]:
            nmck_value = parse_price(str(record[field]))
            break
    
    if nmck_value > 0:
        # Создаем диапазон ±20% от НМЦК
        params["nmck_range"] = (nmck_value * 0.8, nmck_value * 1.2)
    
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Поиск схожих закупок и формирование аналитической сводки"
    )
    parser.add_argument(
        "input_file",
        help="Путь к JSON файлу с данными (запрос или история + запрос)"
    )
    parser.add_argument(
        "--history",
        help="Путь к JSON файлу с историческими данными (опционально)",
        default=None
    )
    parser.add_argument(
        "--output",
        help="Путь к файлу для сохранения результатов (по умолчанию вывод в консоль)",
        default=None
    )
    parser.add_argument(
        "--period",
        type=int,
        default=3,
        help="Период поиска в годах (по умолчанию 3)"
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.3,
        help="Минимальный порог схожести 0-1 (по умолчанию 0.3)"
    )
    
    args = parser.parse_args()
    
    # Загрузка входных данных
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{args.input_file}' не найден")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Ошибка: Неверный формат JSON в файле '{args.input_file}': {e}")
        sys.exit(1)
    
    # Определение структуры данных
    history_data = []
    query_record = None
    
    # Если файл содержит историю и запрос
    if isinstance(input_data, dict):
        if "history" in input_data:
            history_data = input_data["history"]
        if "query" in input_data:
            query_record = input_data["query"]
        elif "data" in input_data:
            # Альтернативный формат
            history_data = input_data.get("data", [])
            query_record = input_data.get("query")
    
    # Если файл - это просто запись для поиска
    if query_record is None:
        if isinstance(input_data, list):
            # Если список, считаем что это история, а запроса нет
            history_data = input_data
            print("Предупреждение: В файле только история, нет запроса на поиск.")
            print("Добавьте ключ 'query' с записью для поиска.")
            sys.exit(1)
        elif isinstance(input_data, dict):
            query_record = input_data
    
    # Загрузка отдельного файла истории если указан
    if args.history:
        try:
            with open(args.history, 'r', encoding='utf-8') as f:
                hist_data = json.load(f)
                if isinstance(hist_data, list):
                    history_data = hist_data
                elif isinstance(hist_data, dict) and "data" in hist_data:
                    history_data = hist_data["data"]
        except Exception as e:
            print(f"Ошибка при загрузке файла истории: {e}")
            sys.exit(1)
    
    # Извлечение параметров поиска из запроса
    search_params = extract_query_params(query_record)
    
    if not search_params:
        print("Ошибка: Не удалось извлечь параметры поиска из записи.")
        print("Проверьте наличие полей: 'Организация...', 'Регион', 'Наименование объекта...'")
        sys.exit(1)
    
    # Инициализация анализатора
    analyzer = ProcurementAnalyzer()
    
    # Загрузка истории если есть
    if history_data:
        print(f"Загрузка исторических данных: {len(history_data)} записей...")
        analyzer.load_procurements(history_data)
    else:
        print("Предупреждение: Исторические данные не предоставлены. Поиск будет выполнен по пустой базе.")
    
    # Поиск схожих закупок
    print("\nПоиск схожих закупок...")
    print(f"Параметры поиска:")
    for key, value in search_params.items():
        print(f"  - {key}: {value}")
    
    similar = analyzer.find_similar(
        customer=search_params.get("customer", ""),
        region=search_params.get("region", ""),
        work_type=search_params.get("work_type", ""),
        nmck_range=search_params.get("nmck_range"),
        period_years=args.period,
        min_similarity=args.similarity
    )
    
    print(f"\nНайдено схожих закупок: {len(similar)}")
    
    # Формирование сводки
    current_nmck = 0.0
    if "nmck_range" in search_params:
        current_nmck = search_params["nmck_range"][0] / 0.8  # Восстанавливаем исходное значение
    
    summary = analyzer.generate_summary(
        similar_procurements=similar,
        current_customer=search_params.get("customer", ""),
        current_region=search_params.get("region", ""),
        current_work_type=search_params.get("work_type", ""),
        current_nmck=current_nmck
    )
    
    # Подготовка результата
    result = {
        "query": query_record,
        "search_params": search_params,
        "found_count": len(similar),
        "summary": {
            "count": summary.count,
            "customer": summary.customer,
            "region": summary.region,
            "work_type": summary.work_type,
            "nmck": summary.nmck,
            "avg_reduction_percent": round(summary.avg_reduction, 2) if summary.avg_reduction else None,
            "min_reduction_percent": round(summary.min_reduction, 2) if summary.min_reduction else None,
            "max_reduction_percent": round(summary.max_reduction, 2) if summary.max_reduction else None,
            "median_reduction_percent": round(summary.median_reduction, 2) if summary.median_reduction else None
        },
        "similar_procurements": [p.to_dict() for p in similar]
    }
    
    # Вывод результатов
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены в файл: {args.output}")
    else:
        print("\n--- Аналитическая сводка ---")
        print(f"Количество найденных закупок: {summary.count}")
        print(f"Заказчик: {summary.customer}")
        print(f"Регион: {summary.region}")
        print(f"Вид работ: {summary.work_type}")
        print(f"НМЦК: {summary.nmck:.2f}")
        if summary.avg_reduction is not None:
            print(f"\nСтатистика снижения цены:")
            print(f"  Среднее снижение: {summary.avg_reduction:.2f}%")
            print(f"  Минимальное снижение: {summary.min_reduction:.2f}%")
            print(f"  Максимальное снижение: {summary.max_reduction:.2f}%")
            print(f"  Медианное снижение: {summary.median_reduction:.2f}%")
        
        if similar:
            print(f"\n--- Список схожих закупок ({len(similar)}) ---")
            for i, proc in enumerate(similar[:5], 1):  # Показываем первые 5
                print(f"{i}. {proc.work_type[:60]}...")
                print(f"   Заказчик: {proc.customer[:40]}...")
                print(f"   Цена: {proc.nmck:.2f}, Снижение: {proc.reduction_percent}%")
            if len(similar) > 5:
                print(f"... и ещё {len(similar) - 5} закупок")


if __name__ == "__main__":
    main()
