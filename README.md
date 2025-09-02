# Предсказание оттока клиентов телеком-компании

## Описание задачи

Бинарная модель оценивает вероятность `Churn` до запуска кампании удержания. Ошибки имеют разную цену: false negative пропускает уходящего клиента, false positive расходует бюджет.

## Цель проекта

Ранжировать клиентов и выбрать верхние 20% риска для ограниченной кампании. Основная offline-метрика — PR-AUC, порог подбирается только на validation.

## Архитектура решения

Валидация схемы и уникальности `customerID` → стратифицированные train/validation/test →
заполнение, масштабирование и OneHot внутри train-only pipeline → Dummy, Logistic Regression,
Decision Tree, Random Forest и Gradient Boosting → выбор по validation PR-AUC → test-метрики,
FP/FN по сегментам и permutation importance. `customerID` не является признаком.

## Структура каталогов

`src/telco_churn` — схема, модели, генератор и CLI; `tests` — offline workflow; `data` — контракт; `artifacts`/`reports` создаются локально; `.github/workflows` — CI.

## Используемые технологии

Python 3.11, NumPy, pandas, scikit-learn, joblib, pytest, Ruff.

## Требования к окружению

Python 3.11.15; зависимости закреплены в `pyproject.toml`.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Подготовка данных

Контрольный источник — репозиторий [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d),
распространяемый по Apache-2.0. Эта лицензия относится к репозиторию IBM; происхождение и
условия использования конкретного CSV нужно проверять отдельно. Реальные данные и готовое
IBM-решение здесь не включены, поэтому локальные версия CSV, размер и SHA-256 отсутствуют;
их нужно зафиксировать перед полным экспериментом. Для smoke-run используйте `make smoke`.

## Запуск обучения

```bash
telco-train --data data/smoke.csv --artifact artifacts/model.joblib --report reports/validation_metrics.json
```

## Запуск оценки

```bash
telco-evaluate --data data/smoke.csv --artifact artifacts/model.joblib --metrics reports/test_metrics.json --errors reports/test_errors.csv --importance reports/permutation_importance.csv
```

## Запуск инференса

```bash
telco-predict --data data/smoke.csv --artifact artifacts/model.joblib --output reports/predictions.csv
```

## Метрики

PR-AUC и ROC-AUC для ранжирования; precision, recall, F1, confusion matrix и selected
fraction для рабочей очереди. Она содержит ровно `ceil(n × budget_fraction)` клиентов;
равные score разрешаются стабильным исходным порядком. Отчёт дополняют error rate/recall по
`Contract` и `InternetService` и permutation importance на test.

## Тестирование

`make check` запускает Ruff и pytest. Тесты используют только локальную синтетику и не выполняют загрузок.

## Ограничения

Демонстрационная выборка IBM не представляет текущую клиентскую базу. Порог 20% — учебное предположение, а не реальный бизнес-бюджет. Importance не доказывает причинность; перед использованием нужны временный holdout, fairness review и мониторинг дрейфа.

## Полученные результаты

Результаты на реальном IBM-наборе пока не получены и не заявляются. Централизованный запуск может добавить фактические test-метрики; smoke-цифры останутся только техническим сигналом.

## Статус проекта

Инженерная реализация завершена; реальная оценка и фиксация версии датасета ожидают централизованного запуска.
