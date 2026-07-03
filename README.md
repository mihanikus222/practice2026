# Практика БВТ22 — вариант 9

**Репозиторий:** [https://github.com/mihanikus222/practice2026](https://github.com/mihanikus222/practice2026)

Классификация и сегментация дефектов на синтетических данных автомобильной тематики: синтетический датасет, обучение классификаторов и сегментаторов, сравнение архитектур, демо-приложение и метрики.

## Установка

```bash
cd practice-bvt22-variant9
python -m pip install -r requirements.txt
```

## Воспроизведение данных и обучения

Крупные файлы (`data/raw/images/`, `data/raw/masks/`, веса `runs/**/*.pt`) **не хранятся в Git**. После клонирования сгенерируйте данные и запустите пайплайн:

```bash
python data/generate_dataset.py
python scripts/run_all_experiments.py
python scripts/generate_examples.py
```

При необходимости догоните оставшиеся эксперименты:

```bash
python scripts/run_remaining.py
```

## Инференс

```bash
set PYTHONPATH=src
python src/inference.py data/raw/images/sample_0000.png
```

## Артефакты

- `report/comparison_table.csv`, `report/comparison_table.xlsx` — сводная таблица моделей
- `report/figures/` — графики и примеры для отчёта
- `runs/**/results.json`, `runs/comparison_table.json`, `runs/history.json` — метрики экспериментов

## Структура

- `data/` — генератор данных, `metadata.json`, `stats.json` (изображения и маски — локально)
- `src/` — обучение, модели, инференс
- `scripts/` — полный цикл экспериментов и визуализаций
- `configs/` — конфигурации
- `runs/` — метрики и демо-выходы (без `.pt` — локально)
- `report/` — таблицы сравнения и figures

## Вариант 9 — требования

1. Определение классов дефектов (классификация)
2. Выделение областей повреждения (сегментация)
3. Сравнение классификационных и сегментационных подходов
4. Не менее 5 архитектур (реализовано 10: 5 + 5)
5. Метрики: IoU, Dice, pixel accuracy; accuracy, precision, recall, F1
6. Демо-приложение с визуализацией, маской и наложением эффекта
7. Гиперпараметры в JSON, таблица в Excel/CSV
