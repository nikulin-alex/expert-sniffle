"""
Procurement AI - модуль анализа закупок.
"""

from .models import ProcurementRecord, AnalysisSummary
from .analyzer import ProcurementAnalyzer

__version__ = "0.1.0"
__all__ = [
    "ProcurementAnalyzer", 
    "ProcurementRecord", 
    "AnalysisSummary",
]
