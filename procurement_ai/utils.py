"""
Вспомогательные функции для модуля анализа закупок.
"""

import re
from typing import List, Set


def normalize_text(text: str) -> str:
    """Нормализация текста: приведение к нижнему регистру, удаление лишних пробелов."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_keywords(text: str, stop_words: Set[str] = None) -> List[str]:
    """Извлечение ключевых слов из текста."""
    if stop_words is None:
        stop_words = {
            "и", "в", "на", "для", "по", "с", "о", "от", "до", "из",
            "the", "a", "an", "of", "for", "to", "in", "on", "at"
        }
    
    text = normalize_text(text)
    words = text.split()
    
    keywords = []
    for word in words:
        word = re.sub(r"[^\wа-яё]", "", word)
        if word and len(word) > 2 and word not in stop_words:
            keywords.append(word)
    
    return keywords


def calculate_median(values: List[float]) -> float:
    """Расчёт медианы списка чисел."""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    else:
        return sorted_values[mid]


def parse_price(price_str: str) -> float:
    """Парсинг строки с ценой в числовое значение."""
    if not price_str:
        return 0.0
    
    if isinstance(price_str, (int, float)):
        return float(price_str)
    
    price_str = price_str.replace(" ", "").replace(",", ".")
    price_str = re.sub(r"[^\d.]", "", price_str)
    
    try:
        return float(price_str)
    except ValueError:
        return 0.0


def similarity_score(text1: str, text2: str) -> float:
    """
    Расчёт коэффициента схожести двух текстов на основе общих слов.
    Возвращает значение от 0 до 1.
    """
    words1 = set(extract_keywords(text1))
    words2 = set(extract_keywords(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0
