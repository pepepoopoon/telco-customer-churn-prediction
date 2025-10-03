# Серия экспериментов churn-модели

Каталог содержит 72 фактически выполненных синтетических эксперимента: по 18 запусков
для retention budget, learning curve, seed stability и data quality. Каждый JSON создан
отдельным вызовом runner и хранит полную конфигурацию, размеры split, сравнение с prior
baseline, holdout-метрики и diagnostics.

## Воспроизводимый запуск

Runner использует детерминированный генератор, стратифицированные train/validation/test
split и train-only preprocessing. Пример:

```bash
python -m telco_churn.experiment \
  --scenario retention_budget \
  --seed 20250809 \
  --rows 480 \
  --budget-fraction 0.20 \
  --output experiments/results/example.json
```

Поддерживаются сценарии `retention_budget`, `learning_curve`, `seed_stability` и
`data_quality`. Результат имеет `schema_version: 1`; одинаковая конфигурация создаёт
одинаковый JSON.

## Матрица запусков

| Сценарий | Запусков | Варьируемый параметр |
|---|---:|---|
| Retention budget | 18 | Доля очереди от 5% до 100% |
| Learning curve | 18 | Доля доступного train от 15% до 100% |
| Seed stability | 18 | Seed от `20250801` до `20250818` |
| Data quality | 18 | Пропуски и unseen-категории по отдельности и вместе |

## Измеренные результаты

При фиксированных seed `20250809`, 480 строках и неизменном test holdout бюджетная
очередь дала следующие рабочие точки:

| Бюджет | Выбрано клиентов | Найдено churn | Recall | Precision |
|---:|---:|---:|---:|---:|
| 5% | 5 | 2 | 0.105 | 0.400 |
| 20% | 20 | 7 | 0.368 | 0.350 |
| 100% | 96 | 19 | 1.000 | 0.198 |

- learning curve: test PR-AUC менялся от **0.353** до **0.441**; лучший результат в
  этой синтетической серии получен при 50% доступного train;
- seed stability: средний test PR-AUC — **0.404**, стандартное отклонение — **0.095**,
  диапазон — **0.261–0.591**;
- data quality: test PR-AUC находился в диапазоне **0.368–0.402**;
- финальный комбинированный запуск с 10% пропусков и 15% unseen-категорий дал
  PR-AUC **0.379**, ROC-AUC **0.714**, precision **0.350** и recall **0.368**;
- в финальном permutation-отчёте наибольшая средняя важность у `TechSupport` — **0.049**.

Финальный результат находится в
[`087_data_quality_combined_010_015.json`](results/087_data_quality_combined_010_015.json).

## Диагностический контракт

Каждый JSON включает:

- feature-level missing counts и доли строк с пропусками;
- unseen values и затронутые строки для validation/test;
- recall, error rate, selection rate и PR-AUC по `Contract` и `InternetService`;
- permutation importance на test с тремя детерминированными повторениями;
- test-дельту weighted logistic regression относительно prior baseline.

## Ограничения

Все результаты относятся только к синтетическому генератору и не оценивают качество на
IBM Telco или production-данных. Seed-вариативность заметна, а permutation importance не
доказывает причинность. Перед продуктовым решением нужны зафиксированная версия реального
датасета, временной holdout, стоимость контакта и независимая проверка сегментов.
