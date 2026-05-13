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
        okpd2: str = "",
        keywords: List[str] = None,
        nmck_range: Tuple[float, float] = None,
        period_years: int = 0,
        min_similarity: float = 0.3,
        require_auction: bool = False,
        limit: int = 5
    ) -> List[ProcurementRecord]:
        """
        Поиск схожих закупок по заданным критериям.
        
        Args:
            customer: Наименование заказчика для поиска.
            region: Регион для поиска (не используется в расчёте схожести).
            work_type: Вид работ/услуг для поиска.
            okpd2: Код ОКПД2 для поиска.
            keywords: Список ключевых слов для поиска.
            nmck_range: Диапазон НМЦК (min, max).
            period_years: Период поиска в годах (0 = без ограничений).
            min_similarity: Минимальный порог схожести (0-1).
            require_auction: Требовать наличие результатов аукциона.
            limit: Максимальное количество возвращаемых результатов.
        
        Returns:
            Список схожих закупок.
        """
        similar = []
        
        # Вычисляем дату отсечения для периода
        cutoff_date = None
        if period_years > 0:
            cutoff_date = datetime.now() - timedelta(days=period_years * 365)
        
        # Если ни один фильтр не задан, возвращаем все записи (с аукционами или без)
        if not customer and not work_type and not okpd2 and not nmck_range:
            result = []
            for p in self.procurements:
                if require_auction and (not p.auction_results or len(p.auction_results) == 0):
                    continue
                # Применяем фильтр по периоду
                if cutoff_date and p.publication_date and p.publication_date < cutoff_date:
                    continue
                result.append(p)
            return result[:limit]
        
        for proc in self.procurements:
            score = 0.0
            max_score = 0.0
            
            # Проверка по заказчику (вес 10%)
            if customer:
                max_score += 0.1
                if customer.lower() in proc.customer.lower() or proc.customer.lower() in customer.lower():
                    score += 0.1
            
            # Проверка по виду работ (вес 40%)
            if work_type:
                max_score += 0.4
                work_similarity = similarity_score(work_type, proc.work_type)
                score += work_similarity * 0.4
            
            # Проверка по ОКПД2 (вес 10%)
            if okpd2 and proc.okpd2:
                max_score += 0.1
                # Сравниваем первые знаки кода ОКПД2 (группировка)
                okpd2_base = okpd2.replace(".", "")[:6]
                proc_okpd2_base = proc.okpd2.replace(".", "")[:6]
                if okpd2_base == proc_okpd2_base:
                    score += 0.1
                elif okpd2_base[:4] == proc_okpd2_base[:4]:
                    score += 0.05  # Частичное совпадение
            
            # Проверка по диапазону НМЦК (вес 50%)
            if nmck_range:
                max_score += 0.5
                nmck_min, nmck_max = nmck_range
                if nmck_min <= proc.nmck <= nmck_max:
                    # Чем ближе цена к середине диапазона, тем выше оценка
                    range_mid = (nmck_min + nmck_max) / 2
                    range_width = nmck_max - nmck_min
                    if range_width > 0:
                        deviation = abs(proc.nmck - range_mid) / range_width
                        price_score = max(0, 1 - deviation)  # От 0 до 1
                        score += 0.5 * price_score
                    else:
                        score += 0.5
                else:
                    # Если цена вне диапазона, но близко - даем частичный балл
                    distance = 0
                    if proc.nmck < nmck_min:
                        distance = nmck_min - proc.nmck
                    else:
                        distance = proc.nmck - nmck_max
                    
                    # Нормализуем расстояние относительно размера диапазона
                    range_width = nmck_max - nmck_min
                    if range_width > 0 and distance < range_width:
                        partial_score = 0.5 * (1 - distance / range_width)
                        score += partial_score
            
            # Применяем фильтр по периоду
            if cutoff_date and proc.publication_date and proc.publication_date < cutoff_date:
                continue
            
            # Если требуется аукцион, но его нет — пропускаем
            if require_auction and (not proc.auction_results or len(proc.auction_results) == 0):
                continue
            
            # Пропускаем записи без информации о снижении цены (для корректного прогноза)
            if proc.reduction_percent is None:
                continue
            
            # Добавляем запись если схожесть выше порога
            if max_score > 0 and score / max_score >= min_similarity:
                similar.append((proc, score / max_score))
        
        # Сортируем по убыванию схожести и берём top N
        similar.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in similar[:limit]]
    
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
        period_years: int = 0
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
