"""
Основной модуль анализа закупок.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from .models import ProcurementRecord, AnalysisSummary, AuctionInfo
from .utils import calculate_median, normalize_text, similarity_score


class ProcurementAnalyzer:
    """Анализатор закупок для поиска схожих записей и формирования сводок."""
    
    def __init__(self, procurements: List[ProcurementRecord] = None):
        """
        Инициализация анализатора.
        
        Args:
            procurements: Список записей о закупках для анализа.
        """
        self.procurements = procurements or []
    
    def load_procurements(self, procurements: List[Dict[str, Any]]) -> None:
        """
        Загрузка закупок из списка словарей.
        
        Args:
            procurements: Список словарей с данными о закупках.
        """
        self.procurements = [ProcurementRecord.from_dict(p) for p in procurements]
    
    def add_procurement(self, procurement: ProcurementRecord) -> None:
        """Добавление одной закупки в коллекцию."""
        self.procurements.append(procurement)
    
    def find_similar(
        self,
        customer: str = "",
        region: str = "",
        work_type: str = "",
        keywords: List[str] = None,
        nmck_range: Tuple[float, float] = None,
        period_years: int = 3,
        min_similarity: float = 0.3
    ) -> List[ProcurementRecord]:
        """
        Поиск схожих закупок по заданным критериям.
        
        Args:
            customer: Наименование заказчика для поиска.
            region: Регион для поиска.
            work_type: Вид работ/услуг для поиска.
            keywords: Список ключевых слов для поиска.
            nmck_range: Диапазон НМЦК (min, max).
            period_years: Период поиска в годах.
            min_similarity: Минимальный порог схожести (0-1).
        
        Returns:
            Список схожих закупок.
        """
        similar = []
        
        # Если ни один фильтр не задан, возвращаем все записи с аукционами
        if not customer and not region and not work_type and not nmck_range:
            return [p for p in self.procurements if p.auction_results and len(p.auction_results) > 0]
        
        for proc in self.procurements:
            score = 0.0
            max_score = 0.0
            
            # Проверка по заказчику (вес 30%)
            if customer:
                max_score += 0.3
                if customer.lower() in proc.customer.lower() or proc.customer.lower() in customer.lower():
                    score += 0.3
            
            # Проверка по региону (вес 25%)
            if region:
                max_score += 0.25
                if region.lower() == proc.region.lower():
                    score += 0.25
            
            # Проверка по виду работ (вес 25%)
            if work_type:
                max_score += 0.25
                work_similarity = similarity_score(work_type, proc.work_type)
                score += work_similarity * 0.25
            
            # Проверка по диапазону НМЦК (вес 10%)
            if nmck_range:
                max_score += 0.1
                nmck_min, nmck_max = nmck_range
                if nmck_min <= proc.nmck <= nmck_max:
                    score += 0.1
            
            # Пропускаем записи без аукционов (нечего анализировать)
            if not proc.auction_results or len(proc.auction_results) == 0:
                continue
            
            # Добавляем запись если схожесть выше порога
            if max_score > 0 and score / max_score >= min_similarity:
                similar.append(proc)
        
        return similar
    
    def generate_summary(
        self,
        similar_procurements: List[ProcurementRecord],
        current_customer: str = "",
        current_region: str = "",
        current_work_type: str = "",
        current_nmck: float = 0.0
    ) -> AnalysisSummary:
        """
        Формирование аналитической сводки по найденным закупкам.
        
        Args:
            similar_procurements: Список схожих закупок.
            current_customer: Текущий заказчик (для отображения в сводке).
            current_region: Текущий регион (для отображения в сводке).
            current_work_type: Текущий вид работ (для отображения в сводке).
            current_nmck: Текущая НМЦК (для отображения в сводке).
        
        Returns:
            Аналитическая сводка.
        """
        reductions = []
        
        for proc in similar_procurements:
            if proc.reduction_percent is not None:
                reductions.append(proc.reduction_percent)
        
        summary = AnalysisSummary(
            count=len(similar_procurements),
            customer=current_customer,
            region=current_region,
            work_type=current_work_type,
            nmck=current_nmck,
        )
        
        if reductions:
            summary.avg_reduction = sum(reductions) / len(reductions)
            summary.min_reduction = min(reductions)
            summary.max_reduction = max(reductions)
            summary.median_reduction = calculate_median(reductions)
        
        return summary
    
    def analyze(
        self,
        customer: str = "",
        region: str = "",
        work_type: str = "",
        keywords: List[str] = None,
        nmck_range: Tuple[float, float] = None,
        period_years: int = 3
    ) -> AnalysisSummary:
        """
        Полный цикл анализа: поиск схожих закупок и формирование сводки.
        
        Args:
            customer: Наименование заказчика для поиска.
            region: Регион для поиска.
            work_type: Вид работ/услуг для поиска.
            keywords: Список ключевых слов для поиска.
            nmck_range: Диапазон НМЦК (min, max).
            period_years: Период поиска в годах.
        
        Returns:
            Аналитическая сводка.
        """
        similar = self.find_similar(
            customer=customer,
            region=region,
            work_type=work_type,
            keywords=keywords,
            nmck_range=nmck_range,
            period_years=period_years
        )
        
        return self.generate_summary(
            similar_procurements=similar,
            current_customer=customer,
            current_region=region,
            current_work_type=work_type,
            current_nmck=nmck_range[0] if nmck_range else 0.0
        )
