"""
Модели данных для модуля анализа закупок.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class AuctionInfo:
    """Информация об аукционе."""
    id: str = ""
    status: str = ""
    amount: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuctionInfo":
        """Создание объекта из словаря."""
        amount = data.get("amount", "0")
        if isinstance(amount, str):
            amount = float(amount.replace(" ", "").replace(",", "."))
        return cls(
            id=data.get("id", ""),
            status=data.get("status", ""),
            amount=amount
        )


@dataclass
class ProcurementRecord:
    """Запись о закупке."""
    reg_number: str = ""
    customer: str = ""
    region: str = ""
    work_type: str = ""
    nmck: float = 0.0
    auction_results: List[AuctionInfo] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def final_amount(self) -> Optional[float]:
        """Финальная цена контракта по результатам аукциона."""
        if self.auction_results and self.auction_results[0].amount > 0:
            return self.auction_results[0].amount
        return None
    
    @property
    def reduction_percent(self) -> Optional[float]:
        """Процент снижения цены."""
        if self.final_amount and self.nmck > 0:
            return ((self.nmck - self.final_amount) / self.nmck) * 100
        return None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcurementRecord":
        """Создание объекта из словаря."""
        # Извлечение НМЦК
        nmck_raw = data.get("Начальная (максимальная) цена контракта", "0")
        if isinstance(nmck_raw, str):
            nmck = float(nmck_raw.replace(" ", "").replace(",", "."))
        else:
            nmck = float(nmck_raw) if nmck_raw else 0.0
        
        # Извлечение аукциона
        auction_raw = data.get("auction", [])
        auction_results = [AuctionInfo.from_dict(a) for a in auction_raw] if auction_raw else []
        
        return cls(
            reg_number=data.get("reg_number", ""),
            customer=data.get("Организация, осуществляющая размещение", ""),
            region=data.get("Регион", ""),
            work_type=data.get("Наименование объекта закупки", ""),
            nmck=nmck,
            auction_results=auction_results,
            raw_data=data
        )


@dataclass
class AnalysisSummary:
    """Аналитическая сводка по схожим закупкам."""
    count: int = 0
    customer: str = ""
    region: str = ""
    work_type: str = ""
    nmck: float = 0.0
    avg_reduction: Optional[float] = None
    min_reduction: Optional[float] = None
    max_reduction: Optional[float] = None
    median_reduction: Optional[float] = None
    
    def __str__(self) -> str:
        """Строковое представление сводки."""
        lines = [
            "=" * 50,
            "АНАЛИТИЧЕСКАЯ СВОДКА ПО СХОЖИМ ЗАКУПКАМ",
            "=" * 50,
            f"Количество найденных закупок: {self.count}",
            f"Заказчик: {self.customer}",
            f"Регион: {self.region}",
            f"Вид работ: {self.work_type}",
            f"НМЦК: {self.nmck:,.2f} RUB",
            "-" * 50,
            "СТАТИСТИКА СНИЖЕНИЯ ЦЕНЫ:",
        ]
        
        if self.avg_reduction is not None:
            lines.append(f"  Среднее снижение: {self.avg_reduction:.2f}%")
        else:
            lines.append("  Среднее снижение: нет данных")
            
        if self.min_reduction is not None:
            lines.append(f"  Минимальное снижение: {self.min_reduction:.2f}%")
        else:
            lines.append("  Минимальное снижение: нет данных")
            
        if self.max_reduction is not None:
            lines.append(f"  Максимальное снижение: {self.max_reduction:.2f}%")
        else:
            lines.append("  Максимальное снижение: нет данных")
            
        if self.median_reduction is not None:
            lines.append(f"  Медианное снижение: {self.median_reduction:.2f}%")
        else:
            lines.append("  Медианное снижение: нет данных")
            
        lines.append("=" * 50)
        return "\n".join(lines)
