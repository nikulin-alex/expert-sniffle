# AI Procurement Analysis Module

## Описание
Модуль для анализа исторических данных о закупках, формирования аналитических сводок по схожим закупкам и предсказания стратегий снижения цены.

## Функциональность

### 1. Анализ схожих закупок
Система подбирает историю схожих закупок за заданный период по признакам:
- заказчик
- регион
- вид работ
- ключевые слова
- диапазон НМЦК (начальная максимальная цена контракта)

На основе найденной истории формируется аналитическая сводка:
- количество найденных схожих закупок
- заказчик
- регион
- вид работ
- НМЦК
- средний процент снижения цены
- минимальное, максимальное и медианное снижение

### 2. ML-модель предсказания снижения
- Обучение на исторических данных с известными процентами снижения
- Извлечение признаков из описания закупки (текст, регион, цена, ограничения)
- Предсказание ожидаемого процента снижения цены
- Генерация стратегии участия (минимальное/умеренное/активное/агрессивное снижение)
- Персонализированные рекомендации для каждой закупки
- Сохранение и загрузка обученной модели

## Структура проекта
```
/workspace
├── readme              # Этот файл
├── agents.md           # Контекст разработки
├── procurement_ai/
│   ├── __init__.py     # Экспорт основных классов
│   ├── analyzer.py     # Основной модуль анализа
│   ├── models.py       # Модели данных
│   ├── utils.py        # Вспомогательные функции
│   └── ml_model.py     # ML-модель предсказания снижения
├── train_model.py      # Скрипт обучения модели
└── find_similar.py     # Скрипт поиска схожих закупок
```

## Установка
```bash
pip install -e .
```

## Использование

### Поиск схожих закупок (CLI)
Для поиска схожих закупок и формирования сводки используйте скрипт `find_similar.py`:

```bash
# Базовый запуск (запись для поиска в input.json)
python find_similar.py input.json

# С сохранением результатов в файл
python find_similar.py input.json --output result.json

# С указанием отдельного файла истории
python find_similar.py query.json --history history.json --output result.json

# С настройкой периода и порога схожести
python find_similar.py input.json --period 5 --similarity 0.5
```

**Формат входного файла (JSON):**

1. **Только запрос:**
```json
{
    "Организация, осуществляющая размещение": "МУНИЦИПАЛЬНОЕ...",
    "Регион": "Свердловская обл",
    "Наименование объекта закупки": "устройство площадки...",
    "Начальная (максимальная) цена контракта": "337 030,00"
}
```

2. **Запрос + история в одном файле:**
```json
{
    "history": [...массив исторических закупок...],
    "query": {...запись для поиска...}
}
```

**Выходные данные:**
- Количество найденных закупок
- Статистика снижения (среднее, мин/макс, медиана)
- Список найденных схожих закупок

---

### Анализ схожих закупок (Python API)
```python
from procurement_ai import ProcurementAnalyzer

analyzer = ProcurementAnalyzer()
analyzer.load_procurements(historical_data)  # загрузка данных
result = analyzer.find_similar(
    customer="МУНИЦИПАЛЬНОЕ КАЗЕННОЕ УЧРЕЖДЕНИЕ...",
    region="Свердловская обл",
    work_type="устройство площадки",
    keywords=["собаки", "выгул"],
    nmck_range=(300000, 400000),
    period_years=3
)
summary = analyzer.generate_summary(result)
print(summary)
```

### Предсказание стратегии снижения
```python
from procurement_ai import ReductionStrategyPredictor

# Создание и обучение модели
predictor = ReductionStrategyPredictor()
predictor.train(historical_data)  # данные с полем 'reduction_percent' или 'auction'

# Предсказание для новой закупки
result = predictor.predict(new_record)
print(f"Предсказанное снижение: {result['predicted_reduction_percent']:.2f}%")
print(f"Стратегия: {result['strategy']}")
print(f"Рекомендации: {result['recommendations']}")
```

### Комбинированный анализ (рекомендуется)
```python
# 1. Найти схожие закупки
similar = analyzer.find_similar(customer="...", region="...", work_type="...")
stats = analyzer.generate_summary(similar)

# 2. Обучить ML-модель на исторических данных
predictor.train(historical_data)

# 3. Получить предсказание для новой закупки
ml_result = predictor.predict(new_record)

# 4. Использовать оба источника для принятия решения
print(f"Историческое среднее снижение: {stats.avg_reduction:.2f}%")
print(f"ML предсказание: {ml_result['predicted_reduction_percent']:.2f}%")
```

### Сохранение и загрузка модели
```python
# Сохранение
predictor.save_model('model.json')

# Загрузка
predictor = ReductionStrategyPredictor()
predictor.load_model('model.json')
```

## Запуск тестов
Тесты удалены для минимизации избыточного кода. Рекомендуется ручное тестирование через CLI-скрипты.

## Зависимости
- Python 3.8+
- Стандартная библиотека (без внешних зависимостей)

## Обучение модели
Для обучения модели на исторических данных используйте скрипт `train_model.py`:

```bash
# Базовое обучение с сохранением в procurement_ai/model.json
python train_model.py history.json

# Обучение с указанием пути сохранения модели
python train_model.py history.json my_models/best_model.json
```

Скрипт загружает данные из JSON файла, обучает модель и сохраняет веса для дальнейшего использования.

## Резюме этапов разработки

### Этап 1: Базовый анализ и поиск
Реализован модуль поиска схожих закупок и формирования статистической сводки.

**Возможности:**
- Поиск по заказчику, региону, виду работ, ключевым словам и диапазону НМЦК.
- Расчет статистики снижения цены (среднее, мин/макс, медиана).
- Фильтрация по временному периоду.

**Использование:**
```python
from procurement_ai import ProcurementAnalyzer

analyzer = ProcurementAnalyzer()
analyzer.load_procurements(historical_data)

# Поиск
similar = analyzer.find_similar(
    customer="МУНИЦИПАЛЬНОЕ...",
    region="Свердловская обл",
    work_type="устройство площадки",
    nmck_range=(300000, 400000),
    period_years=3
)

# Сводка
stats = analyzer.generate_summary(similar)
print(f"Найдено: {stats.count}, Среднее снижение: {stats.avg_reduction}%")
```

---

### Этап 2: ML-модель и предсказание стратегий + CLI скрипты
Добавлен модуль машинного обучения для предсказания оптимальной стратегии снижения цены, а также CLI-скрипты для удобной работы.

**Возможности:**
- **Извлечение признаков:** Текст описания (TF-IDF стиль), регион, тип закупки, логарифм НМЦК, наличие обеспечения.
- **Алгоритм:** Линейная регрессия с градиентным спуском и L2-регуляризацией (без внешних зависимостей).
- **Стратегии:** Классификация на "Минимальное", "Умеренное", "Активное" и "Агрессивное" снижение.
- **Сохранение:** Модель сохраняется в JSON для использования в продакшене.
- **CLI скрипты:** `train_model.py` для обучения, `find_similar.py` для поиска.

**Обучение модели:**
```bash
python train_model.py history.json procurement_ai/model.json
```

**Поиск схожих закупок:**
```bash
python find_similar.py query.json --history history.json --output result.json
```

**Предсказание:**
```python
from procurement_ai import ReductionStrategyPredictor

predictor = ReductionStrategyPredictor()
predictor.load_model("procurement_ai/model.json")

result = predictor.predict(new_record)
print(f"Прогноз снижения: {result['predicted_reduction_percent']:.2f}%")
print(f"Рекомендуемая стратегия: {result['strategy']}")
```

---

### Этап 3: Комплексный анализ (Планируется)
*На данном этапе планируется объединение статистического подхода и ML-предсказаний для формирования итоговой рекомендации с оценкой уверенности.*
