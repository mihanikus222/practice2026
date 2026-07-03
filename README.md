# Практика БВТ22 — вариант 9

**Репозиторий:** [https://github.com/mihanikus222/practice2026](https://github.com/mihanikus222/practice2026)

Обнаружение и сегментация трещин на бетонных поверхностях: синтетический датасет, обучение классификатора и сегментатора, сравнение архитектур, демо-приложение и отчёт.

## Установка

```bash
cd practice-bvt22-variant9
python -m pip install -r requirements.txt
```

## Воспроизведение данных и обучения

Крупные файлы (`data/raw/images/`, `data/raw/masks/`, веса `runs/**/*.pt`) **не хранятся в Git**. После клонирования сгенерируйте датасет и заново обучите модели:

```bash
python data/generate_dataset.py
python scripts/run_all_experiments.py
python scripts/generate_examples.py
```

При необходимости дозапустите оставшиеся эксперименты:

```bash
python scripts/run_remaining.py
```

## Инференс

```bash
set PYTHONPATH=src
python src/inference.py data/raw/images/sample_0000.png
```

## Отчёт

Перед генерацией укажите ФИО и группу в `scripts/generate_report.py` (`STUDENT_FIO`, `STUDENT_GROUP`). Ссылка на репозиторий уже задана в `REPO_URL`.

```bash
python scripts/generate_report.py
```

Результат: `report/` (таблицы сравнения, DOCX).

## Структура

- `data/` — генерация датасета, `metadata.json`, `stats.json` (изображения и маски — локально)
- `src/` — обучение, модели, инференс
- `scripts/` — полный цикл экспериментов и отчёт
- `configs/` — конфигурации
- `runs/` — метрики и история (веса `.pt` — локально)
- `report/` — таблицы и DOCX-отчёт

## Вариант 9 — требования

1. Определение наличия трещины (классификация)
2. Выделение области повреждения (сегментация)
3. Сравнение классификационного и сегментационного подходов
4. Не менее 5 архитектур (реализовано 10: 5 + 5)
5. Метрики: IoU, Dice, pixel accuracy; accuracy, precision, recall, F1
6. Демо-приложение с визуализацией, маской и площадью дефекта
7. Гиперпараметры в JSON, отчёт в Excel/DOCX

## Перед сдачей

- ФИО и группа в `scripts/generate_report.py`
- Ссылка на GitHub — в `REPO_URL` (уже указана)
- Дополнительно: даты, корректировки, план (шаблоны в `extracted/`)
