# Локальный MLflow: tracking и Registry

## Что появилось

MLflow не улучшает качество модели. Он отвечает на инженерные вопросы: что запускали, на каких данных, с какими параметрами, какой получили результат, где лежит модель и какая версия сейчас считается основной.

- **Experiment** — группа запусков одной ML-задачи. В проекте это `crypto_direction_classification`.
- **Run** — один конкретный запуск модели с определёнными параметрами и результатами.
- **Parameter** — настройка или неизменяемая характеристика запуска, например `learning_rate=0.05`.
- **Metric** — численный результат, например `validation_macro_f1=0.395529`.
- **Artifact** — относящийся к run файл: manifest признаков, confusion matrix, metadata, CSV поиска или модель.
- **Model Registry** — каталог управляемых версий моделей.
- **Model Version** — конкретная зарегистрированная модель. Номера выдаёт MLflow, код их не хардкодит.
- **Champion** — текущий основной кандидат по зафиксированному критерию.
- **Challenger** — кандидат на возможную замену champion.

## Локальная архитектура

`train_models.py` и MLflow import script отправляют данные на `http://127.0.0.1:5000`. Tracking Server записывает metadata в SQLite `data/mlflow/mlflow.db`, а run artifacts — в локальный `mlartifacts/`. Legacy-каталог `mlruns/` основным backend не является.

Запускать из корня repository:

~~~powershell
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri sqlite:///data/mlflow/mlflow.db `
  --artifacts-destination ./mlartifacts `
  --host 127.0.0.1 `
  --port 5000
~~~

Относительный SQLite URI корректен, пока текущим каталогом является корень проекта. UI: http://127.0.0.1:5000.

Конфигурация находится только в `config.yaml`:

~~~yaml
mlflow:
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "crypto_direction_classification"
  registered_model_name: "crypto_direction_classifier"
  artifact_location: null
  enabled: true
~~~

`artifact_location: null` означает, что artifact destination задаёт сервер. При `enabled: false` tracker становится no-op: training и tuning не требуют сервера. При `enabled: true` выполняется короткий `/health` check до дорогой работы; недоступность сервера не скрывается.

## Runs и воспроизводимость

Каждая baseline-модель создаёт отдельный run: `baseline_dummy`, `baseline_logistic_regression`, `baseline_catboost`, `baseline_xgboost`, `baseline_lightgbm`. В нём есть параметры модели, размер и период Train/Validation, настройки target, aggregate и BUY/HOLD/SELL metrics, training time, feature manifest, evaluation artifacts и подписанная MLflow pyfunc model.

Модель принимает ровно признаки из `feature_manifest.json` и возвращает текстовые `BUY`, `HOLD` или `SELL`. Signature и input example выводятся из реальных строк Validation без `target` и `future_return`. После логирования pipeline загружает MLflow model и сверяет её predictions с исходной моделью.

Для каждого run рассчитываются SHA-256:

- `train_dataset_sha256`;
- `validation_dataset_sha256`;
- `feature_manifest_sha256`.

Также tags содержат Git branch/commit и фактическую версию MLflow. Если Git недоступен, используется `unknown`. Секреты, tokens, API keys и содержимое environment в MLflow не логируются. Test dataset не читается и не используется.

Prediction timestamp сохраняет семантику ML dataset: признаки строки `t` рассчитаны по полностью закрытой свече `t`; прогноз выполняется только после её закрытия. Цель — изменение `Close[t + target.horizon_hours]` относительно `Close[t]`.

## Tuning history без повторного поиска

Полный RandomizedSearchCV повторно не запускается. После baseline training выполните:

~~~powershell
.\.venv\Scripts\python.exe scripts\import_tuning_runs_to_mlflow.py
~~~

Script читает реальные `best_params.json`, CV summaries, полные search CSV, validation artifacts и tuned joblib-модели этапа 4. Runs имеют tag `run_origin=imported_existing_experiment`, поэтому история не выдаётся за изначально отслеживаемый MLflow search. Для будущего запуска `tune_models.py` те же данные будут логироваться напрямую с `run_origin=direct_tuning`.

У tuned runs отдельно видны `cv_mean_macro_f1`, `cv_std_macro_f1`, `validation_macro_f1`, `search_time_seconds`, `n_splits=5`, `gap=3`, `search_n_iter=20` и `scoring=f1_macro`.

Фактические backtest metrics этапа 5 читаются из `artifacts/backtesting/backtest_summary.json`, а не записаны в Python. Они логируются с prefix `backtest_`. Buy & Hold создаётся отдельным run, чтобы не смешивать рыночный benchmark с ML-качеством.

## Registry и aliases

Registry содержит одну сущность `crypto_direction_classifier` и несколько автоматически пронумерованных versions. Минимум регистрируются:

- baseline LightGBM этапа 3 — alias `champion`;
- tuned LightGBM этапа 4 — alias `challenger`.

Baseline имеет Validation Macro F1 `0.395529`, tuned — `0.395198`. Поэтому новая tuned-модель не становится champion только из-за новизны. Latest version не обязательно является best version. Alias меняется только явным вызовом `set_champion(version)`; автоматического promotion rule нет.

Обе версии имеют отрицательную backtest-доходность. Tag `backtest_status=failed_profitability_check` подчёркивает: champion здесь означает лучший зарегистрированный ML-кандидат по Validation Macro F1, а не модель для торговли реальными деньгами.

Проверка Registry и inference:

~~~powershell
.\.venv\Scripts\python.exe scripts\check_mlflow_registry.py
~~~

Script показывает реальные champion/challenger versions и загружает:

~~~text
models:/crypto_direction_classifier@champion
~~~

Затем выполняет predictions на пяти строках Validation. Будущий inference-код сможет сменить фактическую version через alias без изменения URI.

## Проверка UI

В http://127.0.0.1:5000 проверьте:

1. experiment `crypto_direction_classification`;
2. отдельные baseline, tuned и Buy & Hold runs;
3. параметры, validation/class/backtest metrics и время;
4. dataset, evaluation, tuning, backtesting и model artifacts;
5. registered model `crypto_direction_classifier`;
6. model versions, version descriptions/tags и aliases `champion`/`challenger`.

Автоматические тесты проверяют tracking и Registry через API; UI служит дополнительной визуальной проверкой.

## Ограничения этапа

MLflow работает только локально: SQLite backend и filesystem artifacts. Нет PostgreSQL, S3/MinIO, Docker, remote deployment, production serving, FastAPI, автоматического promotion, monitoring или торговли реальными средствами.
