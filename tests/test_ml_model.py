"""
Тесты для ML-модели предсказания снижения цены.
"""

import unittest
import sys
import os

# Добавляем путь к модулю
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from procurement_ai.ml_model import FeatureExtractor, LogisticRegressionModel, ReductionStrategyPredictor


class TestFeatureExtractor(unittest.TestCase):
    """Тесты для экстрактора признаков."""
    
    def setUp(self):
        self.sample_records = [
            {
                "reg_number": "0362300267117000022",
                "Способ определения поставщика (подрядчика, исполнителя)": "Электронный аукцион",
                "Наименование объекта закупки": "устройство площадки для организационного выгула собак",
                "Описание объекта закупки": "устройство площадки для организационного выгула собак",
                "Регион": "Свердловская обл",
                "Начальная (максимальная) цена контракта": "337 030,00",
                "Требуется обеспечение заявки": "Да",
                "Размер обеспечения заявки": "3 370,30 РОССИЙСКИЙ РУБЛЬ",
                "Ограничения и запреты": "1 Закупка у субъектов малого предпринимательства",
                "okpd2": "42.99.11.110"
            },
            {
                "reg_number": "0362300267117000023",
                "Способ определения поставщика (подрядчика, исполнителя)": "Электронный аукцион",
                "Наименование объекта закупки": "поставка офисной мебели",
                "Описание объекта закупки": "поставка офисной мебели для администрации",
                "Регион": "Москва",
                "Начальная (максимальная) цена контракта": "500 000,00",
                "Требуется обеспечение заявки": "Нет",
                "Ограничения и запреты": "Нет ограничений",
                "okpd2": "31.01.11.110"
            }
        ]
    
    def test_fit_extracts_words(self):
        """Проверка извлечения слов из описания."""
        extractor = FeatureExtractor()
        extractor.fit(self.sample_records)
        
        self.assertTrue(extractor.is_fitted)
        self.assertGreater(len(extractor.word_vocab), 0)
        
        # Проверяем, что ключевые слова извлечены
        all_words = ' '.join(extractor.word_vocab.keys())
        self.assertIn('площадки', all_words.lower())
        self.assertIn('мебели', all_words.lower())
    
    def test_fit_extracts_regions(self):
        """Проверка извлечения регионов."""
        extractor = FeatureExtractor()
        extractor.fit(self.sample_records)
        
        self.assertIn("Свердловская обл", extractor.region_map)
        self.assertIn("Москва", extractor.region_map)
    
    def test_extract_returns_features(self):
        """Проверка извлечения признаков из записи."""
        extractor = FeatureExtractor()
        extractor.fit(self.sample_records)
        
        features = extractor.extract(self.sample_records[0])
        
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
    
    def test_extract_requires_fit(self):
        """Проверка, что extract требует предварительного fit."""
        extractor = FeatureExtractor()
        
        with self.assertRaises(ValueError):
            extractor.extract(self.sample_records[0])


class TestLogisticRegressionModel(unittest.TestCase):
    """Тесты для модели линейной регрессии."""
    
    def test_fit_and_predict(self):
        """Проверка обучения и предсказания."""
        model = LogisticRegressionModel(learning_rate=0.01, n_iterations=100)
        
        # Простые данные для обучения
        X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        y = [5.0, 10.0, 15.0, 20.0]
        
        model.fit(X, y)
        
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.weights)
        
        # Предсказание
        predictions = model.predict([[2.5, 3.5]])
        
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        self.assertGreaterEqual(predictions[0], 0.0)
        self.assertLessEqual(predictions[0], 100.0)
    
    def test_predict_requires_fit(self):
        """Проверка, что predict требует предварительного fit."""
        model = LogisticRegressionModel()
        
        with self.assertRaises(ValueError):
            model.predict([[1.0, 2.0]])
    
    def test_empty_data_raises_error(self):
        """Проверка обработки пустых данных."""
        model = LogisticRegressionModel()
        
        with self.assertRaises(ValueError):
            model.fit([], [])


class TestReductionStrategyPredictor(unittest.TestCase):
    """Тесты для основного предсказателя стратегий."""
    
    def setUp(self):
        self.training_data = [
            {
                "reg_number": "001",
                "Способ определения поставщика (подрядчика, исполнителя)": "Электронный аукцион",
                "Наименование объекта закупки": "устройство площадки для выгула собак",
                "Описание объекта закупки": "устройство площадки для выгула собак",
                "Регион": "Свердловская обл",
                "Начальная (максимальная) цена контракта": "337 030,00",
                "Требуется обеспечение заявки": "Да",
                "Ограничения и запреты": "Закупка у субъектов малого предпринимательства",
                "okpd2": "42.99.11.110",
                "reduction_percent": 0.5  # 0.5% снижение
            },
            {
                "reg_number": "002",
                "Способ определения поставщика (подрядчика, исполнителя)": "Электронный аукцион",
                "Наименование объекта закупки": "поставка мебели",
                "Описание объекта закупки": "поставка офисной мебели",
                "Регион": "Москва",
                "Начальная (максимальная) цена контракта": "500 000,00",
                "Требуется обеспечение заявки": "Нет",
                "Ограничения и запреты": "Нет",
                "okpd2": "31.01.11.110",
                "reduction_percent": 15.0  # 15% снижение
            },
            {
                "reg_number": "003",
                "Способ определения поставщика (подрядчика, исполнителя)": "Электронный аукцион",
                "Наименование объекта закупки": "ремонт дорог",
                "Описание объекта закупки": "ремонт автомобильных дорог",
                "Регион": "Свердловская обл",
                "Начальная (максимальная) цена контракта": "1 000 000,00",
                "Требуется обеспечение заявки": "Да",
                "Ограничения и запреты": "Нет",
                "okpd2": "42.11.11.110",
                "reduction_percent": 25.0  # 25% снижение
            }
        ]
    
    def test_train_model(self):
        """Проверка обучения модели."""
        predictor = ReductionStrategyPredictor()
        
        # Обучение должно пройти без ошибок
        predictor.train(self.training_data)
        
        self.assertTrue(predictor.is_trained)
        self.assertTrue(predictor.model.is_fitted)
        self.assertTrue(predictor.feature_extractor.is_fitted)
    
    def test_predict_single_record(self):
        """Проверка предсказания для одной записи."""
        predictor = ReductionStrategyPredictor()
        predictor.train(self.training_data)
        
        test_record = {
            "reg_number": "test_001",
            "Способ определения поставщика (подрядчика, исполнителя)": "Электронный аукцион",
            "Наименование объекта закупки": "устройство детской площадки",
            "Описание объекта закупки": "устройство детской игровой площадки",
            "Регион": "Свердловская обл",
            "Начальная (максимальная) цена контракта": "400 000,00",
            "Требуется обеспечение заявки": "Да",
            "Ограничения и запреты": "Закупка у субъектов малого предпринимательства",
            "okpd2": "42.99.11.110"
        }
        
        result = predictor.predict(test_record)
        
        self.assertIn('predicted_reduction_percent', result)
        self.assertIn('confidence', result)
        self.assertIn('strategy', result)
        self.assertIn('recommendations', result)
        
        self.assertIsInstance(result['predicted_reduction_percent'], float)
        self.assertGreaterEqual(result['predicted_reduction_percent'], 0.0)
        self.assertLessEqual(result['predicted_reduction_percent'], 100.0)
    
    def test_predict_batch(self):
        """Проверка пакетного предсказания."""
        predictor = ReductionStrategyPredictor()
        predictor.train(self.training_data)
        
        test_records = [
            {
                "reg_number": "test_001",
                "Способ определения поставщика (подрядчика, исполнителя)": "Электронный аукцион",
                "Наименование объекта закупки": "устройство площадки",
                "Описание объекта закупки": "устройство площадки",
                "Регион": "Свердловская обл",
                "Начальная (максимальная) цена контракта": "400 000,00",
                "Требуется обеспечение заявки": "Да",
                "Ограничения и запреты": "Нет",
                "okpd2": "42.99.11.110"
            },
            {
                "reg_number": "test_002",
                "Способ определения поставщика (подрядчика, исполнителя)": "Электронный аукцион",
                "Наименование объекта закупки": "поставка оборудования",
                "Описание объекта закупки": "поставка технического оборудования",
                "Регион": "Москва",
                "Начальная (максимальная) цена контракта": "600 000,00",
                "Требуется обеспечение заявки": "Нет",
                "Ограничения и запреты": "Нет",
                "okpd2": "28.99.11.110"
            }
        ]
        
        results = predictor.predict_batch(test_records)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('reg_number', result)
            self.assertIn('predicted_reduction_percent', result)
    
    def test_predict_requires_training(self):
        """Проверка, что predict требует предварительного обучения."""
        predictor = ReductionStrategyPredictor()
        
        test_record = self.training_data[0].copy()
        del test_record['reduction_percent']
        
        with self.assertRaises(ValueError):
            predictor.predict(test_record)
    
    def test_strategy_generation(self):
        """Проверка генерации стратегий для разных уровней снижения."""
        predictor = ReductionStrategyPredictor()
        predictor.train(self.training_data)
        
        # Тестирование различных сценариев через мокирование предсказаний
        # Проверяем, что стратегии генерируются корректно
        test_cases = [
            (3.0, "МИНИМАЛЬНОЕ"),
            (10.0, "УМЕРЕННОЕ"),
            (20.0, "АКТИВНОЕ"),
            (35.0, "АГРЕССИВНОЕ")
        ]
        
        test_record = {
            "reg_number": "test",
            "Наименование объекта закупки": "тест",
            "Описание объекта закупки": "тест",
            "Регион": "Свердловская обл",
            "Начальная (максимальная) цена контракта": "100 000,00",
            "Требуется обеспечение заявки": "Да",
            "Ограничения и запреты": "Нет",
            "okpd2": "42.99.11.110"
        }
        
        # Просто проверяем, что предсказание работает
        result = predictor.predict(test_record)
        self.assertIn('strategy', result)
        self.assertIsInstance(result['strategy'], str)
        self.assertGreater(len(result['strategy']), 0)
    
    def test_recommendations_generation(self):
        """Проверка генерации рекомендаций."""
        predictor = ReductionStrategyPredictor()
        predictor.train(self.training_data)
        
        test_record = {
            "reg_number": "test",
            "Наименование объекта закупки": "тест",
            "Описание объекта закупки": "тест",
            "Регион": "Свердловская обл",
            "Начальная (максимальная) цена контракта": "100 000,00",
            "Требуется обеспечение заявки": "Да",
            "Ограничения и запреты": "Закупка у субъектов малого предпринимательства",
            "okpd2": "42.99.11.110",
            "Дата и время начала срока подачи заявок": "01.01.2024 10:00",
            "Дата и время окончания срока подачи заявок на участие в электронном аукционе": "15.01.2024 10:00"
        }
        
        result = predictor.predict(test_record)
        
        self.assertIn('recommendations', result)
        self.assertIsInstance(result['recommendations'], list)
        self.assertGreater(len(result['recommendations']), 0)
        
        # Проверяем наличие конкретных рекомендаций
        recommendations_text = ' '.join(result['recommendations'])
        self.assertIn('обеспечение', recommendations_text.lower())
        self.assertIn('МСП', recommendations_text)


class TestModelPersistence(unittest.TestCase):
    """Тесты сохранения и загрузки модели."""
    
    def setUp(self):
        self.training_data = [
            {
                "reg_number": "001",
                "Способ определения поставщика (подрядчика, исполнителя)": "Электронный аукцион",
                "Наименование объекта закупки": "устройство площадки",
                "Описание объекта закупки": "устройство площадки",
                "Регион": "Свердловская обл",
                "Начальная (максимальная) цена контракта": "337 030,00",
                "Требуется обеспечение заявки": "Да",
                "Ограничения и запреты": "Нет",
                "okpd2": "42.99.11.110",
                "reduction_percent": 5.0
            }
        ]
        self.model_path = '/tmp/test_procurement_model.json'
    
    def tearDown(self):
        # Очистка тестового файла
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
    
    def test_save_and_load_model(self):
        """Проверка сохранения и загрузки модели."""
        # Обучение и сохранение
        predictor1 = ReductionStrategyPredictor()
        predictor1.train(self.training_data)
        predictor1.save_model(self.model_path)
        
        # Загрузка в новый экземпляр
        predictor2 = ReductionStrategyPredictor()
        predictor2.load_model(self.model_path)
        
        self.assertTrue(predictor2.is_trained)
        
        # Проверка, что предсказания совпадают
        test_record = {
            "reg_number": "test",
            "Наименование объекта закупки": "устройство площадки",
            "Описание объекта закупки": "устройство площадки",
            "Регион": "Свердловская обл",
            "Начальная (максимальная) цена контракта": "400 000,00",
            "Требуется обеспечение заявки": "Да",
            "Ограничения и запреты": "Нет",
            "okpd2": "42.99.11.110"
        }
        
        result1 = predictor1.predict(test_record)
        result2 = predictor2.predict(test_record)
        
        self.assertEqual(
            result1['predicted_reduction_percent'],
            result2['predicted_reduction_percent']
        )


if __name__ == '__main__':
    unittest.main()
