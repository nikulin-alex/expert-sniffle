"""
Модуль извлечения признаков и ML-модели для предсказания снижения цены.
Использует только стандартную библиотеку Python (логистическая регрессия с градиентным спуском).
"""

import math
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


class FeatureExtractor:
    """Извлекает признаки из записи закупки для ML-модели."""
    
    def __init__(self):
        self.word_vocab = {}
        self.region_map = {}
        self.procurement_type_map = {}
        self.is_fitted = False
    
    def fit(self, records: List[Dict[str, Any]]):
        """Построение словарей на основе обучающих данных."""
        all_words = Counter()
        regions = set()
        procurement_types = set()
        
        for record in records:
            # Извлечение слов из описания
            description = record.get('Описание объекта закупки', '') or record.get('Наименование объекта закупки', '')
            words = self._tokenize(description)
            all_words.update(words)
            
            # Регион
            region = record.get('Регион', '')
            if region:
                regions.add(region)
            
            # Способ определения поставщика
            proc_type = record.get('Способ определения поставщика (подрядчика, исполнителя)', '')
            if proc_type:
                procurement_types.add(proc_type)
        
        # Построение словаря слов (топ-500 наиболее частых)
        most_common = all_words.most_common(500)
        self.word_vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # Словари для категориальных признаков
        self.region_map = {region: idx for idx, region in enumerate(sorted(regions))}
        self.procurement_type_map = {pt: idx for idx, pt in enumerate(sorted(procurement_types))}
        
        self.is_fitted = True
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста: нижний регистр, удаление знаков препинания."""
        if not text:
            return []
        text = text.lower()
        # Удаление знаков препинания
        cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        words = cleaned.split()
        # Фильтрация коротких слов и стоп-слов
        stop_words = {'и', 'в', 'во', 'не', 'что', 'как', 'на', 'для', 'от', 'по', 'при', 'из', 'за', 'под', 'над', 'через', 'без', 'про', 'а', 'но', 'или', 'же', 'ли', 'бы', 'т', 'д', 'к', 'м', 'с', 'б', 'г', 'р', 'л', 'ж', 'з', 'ф', 'э', 'ю', 'ц', 'щ', 'ъ', 'ы', 'ь'}
        return [w for w in words if len(w) > 2 and w not in stop_words]
    
    def extract(self, record: Dict[str, Any]) -> List[float]:
        """Извлечение признаков из одной записи."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor должен быть обучен методом fit()")
        
        features = []
        
        # 1. Признаки из описания (Bag of Words)
        description = record.get('Описание объекта закупки', '') or record.get('Наименование объекта закупки', '')
        words = self._tokenize(description)
        word_counts = Counter(words)
        
        for word in sorted(self.word_vocab.keys(), key=lambda x: self.word_vocab[x]):
            features.append(float(word_counts.get(word, 0)))
        
        # 2. Регион (one-hot encoding)
        region = record.get('Регион', '')
        for r in sorted(self.region_map.keys()):
            features.append(1.0 if region == r else 0.0)
        
        # 3. Способ определения поставщика (one-hot encoding)
        proc_type = record.get('Способ определения поставщика (подрядчика, исполнителя)', '')
        for pt in sorted(self.procurement_type_map.keys()):
            features.append(1.0 if proc_type == pt else 0.0)
        
        # 4. НМЦК (нормализованная)
        nmcc_str = record.get('Начальная (максимальная) цена контракта', '0')
        nmcc = self._parse_price(nmcc_str)
        # Логарифмическая нормализация
        features.append(math.log1p(nmcc) / 20.0)  # Деление для масштабирования
        
        # 5. Требуется обеспечение заявки (бинарный)
        ensure_bid = record.get('Требуется обеспечение заявки', 'Нет')
        features.append(1.0 if ensure_bid == 'Да' else 0.0)
        
        # 6. Ограничения для МСП (бинарный)
        restrictions = record.get('Ограничения и запреты', '')
        msp_restriction = 1.0 if 'малого предпринимательства' in restrictions.lower() else 0.0
        features.append(msp_restriction)
        
        # 7. OKPD2 код (первые 2 знака как категория)
        okpd2 = record.get('okpd2', '')
        if okpd2 and len(okpd2) >= 2:
            okpd2_prefix = okpd2[:2]
            try:
                okpd2_num = float(okpd2_prefix)
                features.append(okpd2_num / 100.0)  # Нормализация
            except ValueError:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features
    
    def _parse_price(self, price_str: str) -> float:
        """Парсинг строки цены в число."""
        if not price_str:
            return 0.0
        # Удаление пробелов и замена запятой на точку
        cleaned = price_str.replace(' ', '').replace(',', '.')
        # Удаление валюты и других символов
        cleaned = ''.join(c for c in cleaned if c.isdigit() or c == '.')
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def get_feature_names(self) -> List[str]:
        """Возвращает имена признаков."""
        names = []
        
        # Слова из описания
        for word in sorted(self.word_vocab.keys(), key=lambda x: self.word_vocab[x]):
            names.append(f'word_{word}')
        
        # Регионы
        for region in sorted(self.region_map.keys()):
            names.append(f'region_{region}')
        
        # Способы определения поставщика
        for pt in sorted(self.procurement_type_map.keys()):
            names.append(f'proc_type_{pt}')
        
        names.extend([
            'log_nmcc',
            'require_bid_ensure',
            'msp_restriction',
            'okpd2_prefix'
        ])
        
        return names


class LogisticRegressionModel:
    """Простая модель линейной регрессии с градиентным спуском для предсказания процента снижения."""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, regularization: float = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = 0.0
        self.is_fitted = False
        self.training_history = []
    
    def _sigmoid(self, z: float) -> float:
        """Сигмоидная функция (для ограничения предсказаний)."""
        # Клиппирование для избежания переполнения
        z = max(-500, min(500, z))
        return 1 / (1 + math.exp(-z))
    
    def fit(self, X: List[List[float]], y: List[float]):
        """Обучение модели на данных."""
        if not X or not y:
            raise ValueError("Данные для обучения пусты")
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Инициализация весов
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.training_history = []
        
        for iteration in range(self.n_iterations):
            total_loss = 0.0
            
            # Градиентный спуск
            weight_gradients = [0.0] * n_features
            bias_gradient = 0.0
            
            for i in range(n_samples):
                # Предсказание
                prediction = self._predict_single(X[i])
                
                # Ошибка
                error = prediction - y[i]
                
                # MSE loss
                total_loss += error ** 2
                
                # Градиенты
                for j in range(n_features):
                    weight_gradients[j] += error * X[i][j]
                bias_gradient += error
            
            # Усреднение градиентов и добавление регуляризации
            for j in range(n_features):
                weight_gradients[j] = weight_gradients[j] / n_samples + self.regularization * self.weights[j]
            bias_gradient /= n_samples
            
            # Обновление весов
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * weight_gradients[j]
            self.bias -= self.learning_rate * bias_gradient
            
            avg_loss = total_loss / n_samples
            self.training_history.append(avg_loss)
            
            if iteration % 100 == 0:
                print(f"Итерация {iteration}, Loss: {avg_loss:.6f}")
        
        self.is_fitted = True
    
    def _predict_single(self, x: List[float]) -> float:
        """Предсказание для одного образца."""
        result = self.bias
        for i, xi in enumerate(x):
            result += self.weights[i] * xi
        return result
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Предсказание процента снижения для множества образцов."""
        if not self.is_fitted:
            raise ValueError("Модель должна быть обучена методом fit()")
        
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            # Ограничение предсказания разумными пределами (0-100%)
            pred = max(0.0, min(100.0, pred))
            predictions.append(pred)
        
        return predictions
    
    def predict_with_confidence(self, X: List[List[float]]) -> List[Tuple[float, float]]:
        """Предсказание с оценкой уверенности (на основе дисперсии признаков)."""
        if not self.is_fitted:
            raise ValueError("Модель должна быть обучена методом fit()")
        
        predictions = self.predict(X)
        confidences = []
        
        for i, x in enumerate(X):
            # Простая эвристика для уверенности: чем ближе к среднему значению признаков, тем выше уверенность
            # Вычисляем "норму" вектора признаков как прокси уверенности
            feature_norm = math.sqrt(sum(xi ** 2 for xi in x))
            # Нормализация уверенности (эвристика)
            confidence = min(1.0, feature_norm / 10.0)
            confidences.append((predictions[i], confidence))
        
        return confidences


class ReductionStrategyPredictor:
    """Основной класс для предсказания стратегий снижения цены."""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = LogisticRegressionModel(learning_rate=0.01, n_iterations=500)
        self.is_trained = False
    
    def train(self, records: List[Dict[str, Any]], target_field: str = 'reduction_percent'):
        """
        Обучение модели на исторических данных.
        
        Args:
            records: Список записей закупок с полем процента снижения
            target_field: Имя поля с процентом снижения (по умолчанию 'reduction_percent')
        """
        # Сначала обучаем экстрактор признаков на всех записях
        self.feature_extractor.fit(records)
        
        # Подготовка данных для обучения модели
        X = []
        y = []
        
        for record in records:
            if target_field in record:
                try:
                    features = self.feature_extractor.extract(record)
                    X.append(features)
                    y.append(float(record[target_field]))
                except Exception as e:
                    print(f"Предупреждение: не удалось обработать запись {record.get('reg_number', 'unknown')}: {e}")
        
        if not X:
            raise ValueError("Нет подходящих данных для обучения")
        
        # Обучение модели
        print(f"Обучение модели на {len(X)} записях...")
        self.model.fit(X, y)
        self.is_trained = True
        print("Обучение завершено!")
    
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предсказание процента снижения для новой записи.
        
        Returns:
            Словарь с предсказанным процентом снижения и рекомендациями
        """
        if not self.is_trained:
            raise ValueError("Модель должна быть обучена перед предсказанием")
        
        # Извлечение признаков
        features = self.feature_extractor.extract(record)
        
        # Предсказание
        prediction_result = self.model.predict_with_confidence([features])[0]
        predicted_reduction = prediction_result[0]
        confidence = prediction_result[1]
        
        # Генерация рекомендаций на основе предсказания
        strategy = self._generate_strategy(predicted_reduction, record, confidence)
        
        return {
            'predicted_reduction_percent': round(predicted_reduction, 2),
            'confidence': round(confidence, 2),
            'strategy': strategy,
            'recommendations': self._get_recommendations(predicted_reduction, record)
        }
    
    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Предсказание для множества записей."""
        if not self.is_trained:
            raise ValueError("Модель должна быть обучена перед предсказанием")
        
        results = []
        for record in records:
            result = self.predict(record)
            result['reg_number'] = record.get('reg_number', 'unknown')
            results.append(result)
        
        return results
    
    def _generate_strategy(self, reduction: float, record: Dict[str, Any], confidence: float) -> str:
        """Генерация стратегии на основе предсказанного снижения."""
        nmcc_str = record.get('Начальная (максимальная) цена контракта', '0')
        nmcc = self._parse_price(nmcc_str)
        
        if reduction < 5:
            strategy = "МИНИМАЛЬНОЕ СНИЖЕНИЕ"
            details = f"Ожидаемое снижение менее 5%. Рекомендуется предлагать цену близкую к НМЦК ({nmcc:,.2f} руб.). Конкуренция низкая."
        elif reduction < 15:
            strategy = "УМЕРЕННОЕ СНИЖЕНИЕ"
            details = f"Ожидаемое снижение 5-15%. Оптимальная стратегия - снижение на 8-12% от НМЦК. Средняя конкуренция."
        elif reduction < 30:
            strategy = "АКТИВНОЕ СНИЖЕНИЕ"
            details = f"Ожидаемое снижение 15-30%. Рекомендуется активное участие в аукционе с поэтапным снижением. Высокая конкуренция."
        else:
            strategy = "АГРЕССИВНОЕ СНИЖЕНИЕ"
            details = f"Ожидаемое снижение более 30%. Требуется максимальная готовность к снижению цены. Очень высокая конкуренция или специфика закупки."
        
        if confidence < 0.5:
            details += " (Низкая уверенность предсказания - рекомендуется дополнительный анализ)"
        
        return f"{strategy}: {details}"
    
    def _get_recommendations(self, reduction: float, record: Dict[str, Any]) -> List[str]:
        """Генерация конкретных рекомендаций."""
        recommendations = []
        
        # Проверка обеспечения
        ensure_required = record.get('Требуется обеспечение заявки', 'Нет') == 'Да'
        if ensure_required:
            ensure_amount = record.get('Размер обеспечения заявки', '0')
            recommendations.append(f"Подготовьте обеспечение заявки: {ensure_amount}")
        
        # Проверка МСП
        restrictions = record.get('Ограничения и запреты', '')
        if 'малого предпринимательства' in restrictions.lower():
            recommendations.append("Закупка предназначена для МСП - убедитесь в соответствии требованиям")
        
        # Рекомендации по стратегии
        if reduction < 10:
            recommendations.append("Изучите документацию внимательно - возможно есть специфические требования")
        elif reduction > 25:
            recommendations.append("Будьте готовы к серьезной ценовой конкуренции")
            recommendations.append("Проверьте возможность оптимизации затрат перед участием")
        
        # Срок подачи заявок
        start_date = record.get('Дата и время начала срока подачи заявок', '')
        end_date = record.get('Дата и время окончания срока подачи заявок на участие в электронном аукционе', '')
        if start_date and end_date:
            recommendations.append(f"Период подачи заявок: с {start_date} по {end_date}")
        
        return recommendations
    
    def _parse_price(self, price_str: str) -> float:
        """Парсинг строки цены в число."""
        if not price_str:
            return 0.0
        cleaned = price_str.replace(' ', '').replace(',', '.')
        cleaned = ''.join(c for c in cleaned if c.isdigit() or c == '.')
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def save_model(self, filepath: str):
        """Сохранение модели в файл."""
        model_data = {
            'feature_extractor': {
                'word_vocab': self.feature_extractor.word_vocab,
                'region_map': self.feature_extractor.region_map,
                'procurement_type_map': self.feature_extractor.procurement_type_map,
                'is_fitted': self.feature_extractor.is_fitted
            },
            'model': {
                'weights': self.model.weights,
                'bias': self.model.bias,
                'is_fitted': self.model.is_fitted,
                'learning_rate': self.model.learning_rate,
                'n_iterations': self.model.n_iterations,
                'regularization': self.model.regularization
            },
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str):
        """Загрузка модели из файла."""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.feature_extractor.word_vocab = model_data['feature_extractor']['word_vocab']
        self.feature_extractor.region_map = model_data['feature_extractor']['region_map']
        self.feature_extractor.procurement_type_map = model_data['feature_extractor']['procurement_type_map']
        self.feature_extractor.is_fitted = model_data['feature_extractor']['is_fitted']
        
        self.model.weights = model_data['model']['weights']
        self.model.bias = model_data['model']['bias']
        self.model.is_fitted = model_data['model']['is_fitted']
        self.model.learning_rate = model_data['model']['learning_rate']
        self.model.n_iterations = model_data['model']['n_iterations']
        self.model.regularization = model_data['model']['regularization']
        
        self.is_trained = model_data['is_trained']
        
        print(f"Модель загружена из {filepath}")
