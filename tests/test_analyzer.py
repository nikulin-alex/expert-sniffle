"""
Тесты для модуля анализа закупок.
"""

import unittest
from procurement_ai import ProcurementAnalyzer, ProcurementRecord, AnalysisSummary


class TestProcurementAnalyzer(unittest.TestCase):
    """Тесты для анализатора закупок."""
    
    def setUp(self):
        """Подготовка тестовых данных."""
        self.test_data = [
            {
                "reg_number": "0362300267117000022",
                "Организация, осуществляющая размещение": "МУНИЦИПАЛЬНОЕ КАЗЕННОЕ УЧРЕЖДЕНИЕ \"УПРАВЛЕНИЕ ЖИЛИЩНО-КОММУНАЛЬНОГО ХОЗЯЙСТВА\" ИВДЕЛЬСКОГО ГОРОДСКОГО ОКРУГА",
                "Регион": "Свердловская обл",
                "Наименование объекта закупки": "устройство площадки для организационного выгула собак",
                "Начальная (максимальная) цена контракта": "337 030,00",
                "auction": [
                    {
                        "id": "",
                        "status": "Принято решение",
                        "amount": "335 344,85"
                    }
                ]
            },
            {
                "reg_number": "0362300267117000023",
                "Организация, осуществляющая размещение": "МУНИЦИПАЛЬНОЕ КАЗЕННОЕ УЧРЕЖДЕНИЕ \"УПРАВЛЕНИЕ ЖИЛИЩНО-КОММУНАЛЬНОГО ХОЗЯЙСТВА\" ИВДЕЛЬСКОГО ГОРОДСКОГО ОКРУГА",
                "Регион": "Свердловская обл",
                "Наименование объекта закупки": "благоустройство территории парка",
                "Начальная (максимальная) цена контракта": "500 000,00",
                "auction": [
                    {
                        "id": "",
                        "status": "Завершено",
                        "amount": "450 000,00"
                    }
                ]
            },
            {
                "reg_number": "0362300267117000024",
                "Организация, осуществляющая размещение": "ООО \"СтройСервис\"",
                "Регион": "Москва",
                "Наименование объекта закупки": "ремонт дорожного покрытия",
                "Начальная (максимальная) цена контракта": "1 000 000,00",
                "auction": [
                    {
                        "id": "",
                        "status": "Завершено",
                        "amount": "950 000,00"
                    }
                ]
            }
        ]
        
        self.analyzer = ProcurementAnalyzer()
        self.analyzer.load_procurements(self.test_data)
    
    def test_load_procurements(self):
        """Тест загрузки закупок."""
        self.assertEqual(len(self.analyzer.procurements), 3)
        self.assertIsInstance(self.analyzer.procurements[0], ProcurementRecord)
    
    def test_parse_nmck(self):
        """Тест парсинга НМЦК."""
        record = self.analyzer.procurements[0]
        self.assertAlmostEqual(record.nmck, 337030.00, places=2)
    
    def test_parse_auction_amount(self):
        """Тест парсинга суммы аукциона."""
        record = self.analyzer.procurements[0]
        self.assertIsNotNone(record.final_amount)
        self.assertAlmostEqual(record.final_amount, 335344.85, places=2)
    
    def test_reduction_percent(self):
        """Тест расчёта процента снижения."""
        record = self.analyzer.procurements[0]
        self.assertIsNotNone(record.reduction_percent)
        # (337030 - 335344.85) / 337030 * 100 ≈ 0.5%
        self.assertGreater(record.reduction_percent, 0)
        self.assertLess(record.reduction_percent, 100)
    
    def test_find_similar_by_customer(self):
        """Тест поиска по заказчику."""
        customer = "МУНИЦИПАЛЬНОЕ КАЗЕННОЕ УЧРЕЖДЕНИЕ \"УПРАВЛЕНИЕ ЖИЛИЩНО-КОММУНАЛЬНОГО ХОЗЯЙСТВА\" ИВДЕЛЬСКОГО ГОРОДСКОГО ОКРУГА"
        similar = self.analyzer.find_similar(customer=customer)
        self.assertEqual(len(similar), 2)
    
    def test_find_similar_by_region(self):
        """Тест поиска по региону."""
        similar = self.analyzer.find_similar(region="Свердловская обл")
        self.assertEqual(len(similar), 2)
    
    def test_find_similar_by_work_type(self):
        """Тест поиска по виду работ."""
        similar = self.analyzer.find_similar(work_type="устройство площадки")
        self.assertGreater(len(similar), 0)
    
    def test_find_similar_by_nmck_range(self):
        """Тест поиска по диапазону НМЦК."""
        similar = self.analyzer.find_similar(nmck_range=(300000, 400000))
        self.assertEqual(len(similar), 1)
    
    def test_generate_summary(self):
        """Тест формирования сводки."""
        similar = self.analyzer.find_similar(region="Свердловская обл")
        summary = self.analyzer.generate_summary(
            similar_procurements=similar,
            current_customer="Тестовый заказчик",
            current_region="Свердловская обл",
            current_work_type="Тестовые работы",
            current_nmck=350000.0
        )
        
        self.assertIsInstance(summary, AnalysisSummary)
        self.assertEqual(summary.count, 2)
        self.assertEqual(summary.customer, "Тестовый заказчик")
        self.assertEqual(summary.region, "Свердловская обл")
        self.assertIsNotNone(summary.avg_reduction)
    
    def test_analyze_full_cycle(self):
        """Тест полного цикла анализа."""
        summary = self.analyzer.analyze(
            region="Свердловская обл",
            nmck_range=(300000, 600000)
        )
        
        self.assertIsInstance(summary, AnalysisSummary)
        self.assertGreater(summary.count, 0)
    
    def test_median_calculation(self):
        """Тест расчёта медианы через сводку."""
        # Создаём данные с известными процентами снижения
        test_data = [
            {
                "reg_number": f"test{i}",
                "Организация, осуществляющая размещение": "Test Org",
                "Регион": "Test Region",
                "Наименование объекта закупки": "test work",
                "Начальная (максимальная) цена контракта": "100000",
                "auction": [{"amount": str(100000 - i * 1000)}]
            }
            for i in range(1, 6)  # 1%, 2%, 3%, 4%, 5%
        ]
        
        analyzer = ProcurementAnalyzer()
        analyzer.load_procurements(test_data)
        
        similar = analyzer.find_similar(region="Test Region")
        summary = analyzer.generate_summary(similar, "", "Test Region", "", 100000)
        
        self.assertIsNotNone(summary.median_reduction)
        self.assertAlmostEqual(summary.median_reduction, 3.0, places=1)


if __name__ == "__main__":
    unittest.main()
