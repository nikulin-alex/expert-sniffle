#!/usr/bin/env python3
"""
Точка входа для анализа новой закупки.
Принимает аргументы командной строки и запускает анализ.
"""

import argparse
import json
import sys
from pathlib import Path

from procurement_ai import ProcurementAnalyzer, ProcurementRecord


def load_data(data_path: str) -> list:
    """Загрузка данных о закупках из JSON файла."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Анализ схожих закупок по заданным параметрам"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Путь к JSON файлу с новой закупкой для анализа (параметры берутся из этой записи)"
    )
    
    parser.add_argument(
        "--base", "-b",
        type=str,
        default="data.json",
        help="Путь к JSON файлу с историческими данными для поиска похожих закупок (по умолчанию: data.json)"
    )
    
    parser.add_argument(
        "--customer", "-c",
        type=str,
        default="",
        help="Наименование заказчика для поиска схожих закупок"
    )
    
    parser.add_argument(
        "--region", "-r",
        type=str,
        default="",
        help="Регион для поиска схожих закупок"
    )
    
    parser.add_argument(
        "--work-type", "-w",
        type=str,
        default="",
        help="Вид работ/услуг для поиска схожих закупок"
    )
    
    parser.add_argument(
        "--nmck-min",
        type=float,
        default=None,
        help="Минимальная НМЦК для фильтрации"
    )
    
    parser.add_argument(
        "--nmck-max",
        type=float,
        default=None,
        help="Максимальная НМЦК для фильтрации"
    )
    
    parser.add_argument(
        "--keywords", "-k",
        type=str,
        nargs="+",
        default=None,
        help="Ключевые слова для поиска (через пробел)"
    )
    
    parser.add_argument(
        "--period", "-p",
        type=int,
        default=0,
        help="Период поиска в годах (по умолчанию: 0 - без ограничений по времени)"
    )
    
    parser.add_argument(
        "--min-similarity", "-s",
        type=float,
        default=0.3,
        help="Минимальный порог схожести от 0 до 1 (по умолчанию: 0.3)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Путь к файлу для сохранения результатов (по умолчанию: вывод в консоль)"
    )
    
    args = parser.parse_args()
    
    # Определение файлов для загрузки
    new_procurement_file = None
    base_data_file = Path(args.base)
    
    # Если указан файл с новой закупкой, загружаем его
    if args.data:
        new_procurement_file = Path(args.data)
        if not new_procurement_file.exists():
            print(f"Ошибка: файл с новой закупкой '{args.data}' не найден", file=sys.stderr)
            sys.exit(1)
    
    # Проверка существования файла с базой данных
    if not base_data_file.exists():
        print(f"Ошибка: файл с базой данных '{args.base}' не найден", file=sys.stderr)
        sys.exit(1)
    
    # Загрузка базы исторических данных
    try:
        procurements_data = load_data(args.base)
    except Exception as e:
        print(f"Ошибка при загрузке базы данных: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Инициализация анализатора и загрузка исторических данных
    analyzer = ProcurementAnalyzer()
    analyzer.load_procurements(procurements_data)
    
    # Параметры поиска (по умолчанию пустые)
    customer = args.customer
    region = args.region
    work_type = args.work_type
    nmck_min_val = args.nmck_min
    nmck_max_val = args.nmck_max
    
    # Если передан файл с новой закупкой, извлекаем параметры из него
    if new_procurement_file:
        try:
            new_data = load_data(new_procurement_file)
            if isinstance(new_data, list) and len(new_data) > 0:
                new_record = new_data[0]  # Берем первую запись
            else:
                new_record = new_data
            
            # Извлекаем параметры из новой записи
            if not customer and "Организация, осуществляющая размещение" in new_record:
                customer = new_record["Организация, осуществляющая размещение"]
            if not region and "Регион" in new_record:
                region = new_record["Регион"]
            if not work_type and "Наименование объекта закупки" in new_record:
                work_type = new_record["Наименование объекта закупки"]
            if nmck_min_val is None and "Начальная (максимальная) цена контракта" in new_record:
                nmck_str = new_record["Начальная (максимальная) цена контракта"]
                if isinstance(nmck_str, str):
                    nmck_str = nmck_str.replace(" ", "").replace(",", ".")
                nmck_min_val = float(nmck_str)
        except Exception as e:
            print(f"Ошибка при чтении новой закупки: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Формирование диапазона НМЦК
    nmck_range = None
    if nmck_min_val is not None or nmck_max_val is not None:
        nmck_min = nmck_min_val if nmck_min_val is not None else 0.0
        nmck_max = nmck_max_val if nmck_max_val is not None else float('inf')
        nmck_range = (nmck_min, nmck_max)
    
    # Поиск схожих закупок
    similar = analyzer.find_similar(
        customer=customer,
        region=region,
        work_type=work_type,
        keywords=args.keywords,
        nmck_range=nmck_range,
        period_years=args.period,
        min_similarity=args.min_similarity,
        require_auction=False
    )
    
    # Формирование сводки
    summary = analyzer.generate_summary(
        similar_procurements=similar,
        current_customer=customer or "",
        current_region=region or "",
        current_work_type=work_type or "",
        current_nmck=nmck_range[0] if nmck_range else 0.0
    )
    
    # Подготовка результата
    result = {
        "summary": {
            "count": summary.count,
            "customer": summary.customer,
            "region": summary.region,
            "work_type": summary.work_type,
            "nmck": summary.nmck,
            "avg_reduction": summary.avg_reduction,
            "min_reduction": summary.min_reduction,
            "max_reduction": summary.max_reduction,
            "median_reduction": summary.median_reduction,
        },
        "similar_procurements": [
            {
                "reg_number": p.reg_number,
                "customer": p.customer,
                "region": p.region,
                "work_type": p.work_type,
                "nmck": p.nmck,
                "final_amount": p.final_amount,
                "reduction_percent": p.reduction_percent,
            }
            for p in similar
        ]
    }
    
    # Вывод результатов
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Результаты сохранены в файл: {args.output}")
    else:
        print(str(summary))
        print(f"\nНайдено схожих закупок: {len(similar)}")
        for proc in similar[:10]:  # Показываем первые 10
            print(f"  - {proc.reg_number}: {proc.work_type[:50]}... ({proc.nmck:,.2f} руб.)")
        if len(similar) > 10:
            print(f"  ... и ещё {len(similar) - 10}")


if __name__ == "__main__":
    main()
