# Crypto ML Platform — локальный MLOps pipeline

Учебный production-oriented ML-проект по прогнозированию движения криптовалютного рынка: от загрузки OHLCV и feature engineering до временной валидации, backtesting, MLflow Model Registry и локального FastAPI inference API с PostgreSQL.

## Требования и установка

Проект рассчитан на Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Настройка

Все параметры находятся в `config.yaml`: provider, символ, timeframe, начальная и конечная даты, каталоги хранения, HTTP retry/timeout и уровень логирования. `end_date: null` означает загрузку до текущего доступного момента. Даты без timezone трактуются как UTC.

По умолчанию загружается `BTCUSDT`, `1h`, начиная с `2024-01-01`. Для короткой ручной проверки временно укажите небольшой диапазон, например последние несколько завершённых дней.

## Запуск

```powershell
python scripts/download_market_data.py
```

Другой файл конфигурации можно передать явно:

```powershell
python scripts/download_market_data.py --config .\config.yaml
```

Binance API запрашивается постранично; временные сетевые ошибки, HTTP 429 и 5xx повторяются ограниченное число раз. Provider возвращает единый формат: `timestamp`, `open`, `high`, `low`, `close`, `volume`.

## Результат и повторный запуск

Raw-снимок сохраняется как `data/raw/BTCUSDT_1h_<start>_<end>.parquet`. Он содержит нормализованный ответ provider до очистки. Если файл того же диапазона уже существует, данные безопасно объединяются по timestamp; запись атомарна. Processed-набор атомарно обновляется в `data/processed/BTCUSDT_1h.parquet`; он содержит только шесть канонических колонок, UTC timestamps, уникальные отсортированные свечи и валидные OHLCV-значения. Разрывы временного ряда диагностируются, но не заполняются вымышленными данными.

Большие Parquet-файлы исключены из Git, а каталоги сохранены через `.gitkeep`.

## Тестирование

```powershell
pytest -v
```

Unit-тесты используют fake provider и `httpx.MockTransport`, поэтому не обращаются к реальному Binance API.

## Блок 2 — ML Dataset Preparation

Команда подготовки датасета:

```powershell
python scripts\build_ml_dataset.py
```

Pipeline читает очищенный `data/processed/BTCUSDT_1h.parquet` и рассчитывает только причинные признаки: доходности, SMA, rolling volatility, volume features, характеристики свечи, RSI, MACD и ATR.

### Семантика prediction timestamp

Строка с timestamp `t` описывает свечу `t`. Все её признаки рассчитываются по OHLCV полностью закрытой свечи `t` и более ранних закрытых свечей. Прогноз разрешено выполнять только после закрытия свечи `t`; внутрисвечные, ещё изменяющиеся значения не используются.

Для Binance `timestamp` является UTC-временем открытия свечи. Поэтому при timeframe `1h` строка `12:00` становится доступной для inference после закрытия этой свечи в `13:00`. Timestamp идентифицирует свечу, а не физический момент, когда прогноз уже можно запросить.

Модель прогнозирует изменение Close относительно `close[t]` — цены закрытия текущей свечи — до Close на горизонте `target.horizon_hours`:

```text
future_return[t] = close[t + horizon] / close[t] - 1
```

При `target.horizon_hours: 3` базовой ценой является закрытие свечи `t`, а сравниваемой — закрытие свечи через три часа. По умолчанию будущая доходность от `+0.3%` соответствует `BUY`, до `-0.3%` — `SELL`, промежуточная — `HOLD`. `future_return` сохраняется для анализа, но не входит в явный список features.

Подробный контракт колонок и времени: [docs/ml_dataset.md](docs/ml_dataset.md).

Данные делятся строго хронологически в пропорции 70/15/15. Случайное перемешивание запрещено, поскольку оно смешивает прошлое и будущее и создаёт data leakage. Последние `horizon_hours` строк Train и Validation исключаются (purge), поэтому target предыдущего split не использует цену следующего split.

Создаваемые файлы:

- `data/ml/BTCUSDT_1h_ml_dataset.parquet`;
- `data/ml/BTCUSDT_1h_train.parquet`;
- `data/ml/BTCUSDT_1h_validation.parquet`;
- `data/ml/BTCUSDT_1h_test.parquet`;
- `data/ml/feature_manifest.json`;
- `data/ml/dataset_metadata.json`;
- `data/ml/feature_statistics.json`.

Полная проверка обоих блоков:

```powershell
python -m pytest -v
```

## Блок 3 — Model Training and Validation Comparison

Перед запуском должны быть построены ML-dataset и временные split Блока 2. Обучение запускается командой:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py
```

Pipeline обучает `DummyClassifier`, Logistic Regression, CatBoost, XGBoost и LightGBM на одном и том же Train. Все модели используют только 28 колонок из `data/ml/feature_manifest.json` и оцениваются на одном Validation. Mapping классов един для всех моделей: `SELL=0`, `HOLD=1`, `BUY=2`.

Для Logistic Regression используется sklearn Pipeline `StandardScaler -> LogisticRegression`; scaler обучается только на Train. CatBoost, XGBoost и LightGBM получают признаки без scaling.

Главная метрика выбора — Validation Macro F1. Дополнительно сохраняются Accuracy, Macro Precision, Macro Recall, метрики BUY/HOLD/SELL и confusion matrix. Test dataset зарезервирован для будущей независимой проверки: Блок 3 не читает его, не считает на нём метрики и не использует его для выбора модели или параметров.

Модели сохраняются в `artifacts/models/`, результаты — в `artifacts/model_evaluation/`. Встроенные feature importance бустингов служат только описанием использования признаков конкретной моделью; их шкалы между библиотеками не сопоставимы, и они не доказывают причинное влияние признака на рынок.

Проверка:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Блок 4 — Time-Series CV и подбор гиперпараметров

CatBoost, XGBoost и LightGBM настраиваются командой:

~~~powershell
.\.venv\Scripts\python.exe scripts\tune_models.py
~~~

RandomizedSearchCV использует только Train и пять expanding-window фолдов без перемешивания. Между обучающей и проверочной частями каждого фолда установлен gap, вычисляемый из target.horizon_hours и market_data.timeframe; для горизонта 3 часа и свечей 1h gap равен трём строкам. Метрика поиска — Macro F1 (sklearn f1_macro), число случайных комбинаций — 20, random_state — 42.

Validation загружается только после завершения поиска всех моделей. Лучшие параметры каждой модели затем обучаются на полном Train и сравниваются с результатами этапа 3 на Validation. Test остаётся зарезервированным: pipeline его не читает, не оценивает и не использует для выбора параметров или победителя.

Результаты поиска сохраняются в **artifacts/hyperparameter_search/**, tuned-модели — в **artifacts/tuned_models/**; артефакты этапа 3 не перезаписываются. Подробное описание фолдов, prediction timestamp и файлов: [docs/hyperparameter_tuning.md](docs/hyperparameter_tuning.md).

## Блок 5 — Validation Backtesting

Историческая симуляция запускается без переобучения моделей:

~~~powershell
.\.venv\Scripts\python.exe scripts\run_backtest.py
~~~

Pipeline загружает сохранённые LightGBM этапа 3 и tuned LightGBM этапа 4, заново получает predictions на Validation и сравнивает обе ML-стратегии с Buy & Hold. Сигнал после закрытия свечи t исполняется только по Open свечи t+1. Последний сигнал без следующей свечи игнорируется; открытый LONG в конце принудительно закрывается по последнему Close.

Стратегия работает только в состояниях FLAT/LONG, без short и leverage, с all-in размером позиции. Комиссия и configurable slippage учитываются при исполнении. Сохраняются equity curves, trade logs, доходность, excess return, drawdown, hourly Sharpe, статистика сделок, exposure и fees.

Test dataset не читается. Это виртуальная историческая проверка на Validation, а не реальная торговля и не доказательство будущей прибыльности. Подробности: [docs/backtesting.md](docs/backtesting.md).

## Блок 6 — MLflow Tracking и Model Registry

MLflow 3 используется как дополнительный локальный слой наблюдаемости. Metadata хранится в SQLite `data/mlflow/mlflow.db`, а модели и прочие run artifacts — в `mlartifacts/`. Оба runtime-хранилища исключены из Git.

Сначала из корня проекта запустите сервер в отдельном PowerShell:

~~~powershell
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri sqlite:///data/mlflow/mlflow.db `
  --artifacts-destination ./mlartifacts `
  --host 127.0.0.1 `
  --port 5000
~~~

UI откроется по адресу http://127.0.0.1:5000. Затем создайте пять новых baseline runs:

~~~powershell
.\.venv\Scripts\python.exe scripts\train_models.py
~~~

Импортируйте сохранённые результаты tuning и backtesting без повторного дорогого поиска и создайте Registry:

~~~powershell
.\.venv\Scripts\python.exe scripts\import_tuning_runs_to_mlflow.py
.\.venv\Scripts\python.exe scripts\check_mlflow_registry.py
~~~

Experiment называется `crypto_direction_classification`, единая Registry-модель — `crypto_direction_classifier`. Alias `champion` указывает на baseline LightGBM этапа 3, а `challenger` — на tuned LightGBM этапа 4. Champion означает лучший текущий кандидат по Validation Macro F1, но не готовность к торговле: обе модели имеют отрицательный backtest и помечены `failed_profitability_check`.

Если `mlflow.enabled: false`, обучение и tuning продолжают работать без сервера. Если tracking включён, недоступный сервер обнаруживается до обучения и приводит к понятной ошибке. Полное описание понятий, данных, fingerprints, aliases и ручной проверки: [docs/mlflow.md](docs/mlflow.md).

## Блок 7 — FastAPI inference API и PostgreSQL

API принимает уже рассчитанный вектор из ровно 28 признаков, загружает один раз при старте модель с alias `champion` из MLflow Model Registry, выполняет классификацию `BUY/HOLD/SELL` и сохраняет результат вместе с версией модели в PostgreSQL. API не получает свечи с биржи и не рассчитывает признаки.

Скопируйте пример окружения и задайте настоящий пароль локального пользователя БД:

~~~powershell
Copy-Item .env.example .env
# Отредактируйте DATABASE_URL в .env
~~~

После установки PostgreSQL создайте БД `mlops_pipeline` и пользователя `mlops_user`, затем примените миграцию:

~~~powershell
..venvScriptsalembic.exe upgrade head
..venvScriptspython.exe scriptscheck_database.py
~~~

MLflow Tracking Server должен быть запущен, а alias `champion` — существовать. В отдельном терминале запустите API:

~~~powershell
..venvScriptspython.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
~~~

Swagger UI доступен по адресу http://127.0.0.1:8000/docs. Проверка реального пути от Registry до PostgreSQL:

~~~powershell
..venvScriptspython.exe scriptssmoke_test_api.py
~~~

Основные endpoints: `POST /api/v1/predict`, `GET /api/v1/health/live`, `GET /api/v1/health/ready`, `GET /api/v1/model`, `GET /api/v1/predictions` и `GET /api/v1/predictions/{request_id}`.

Повтор того же запроса с тем же `request_id` возвращает сохранённый ответ с `replayed=true` и не запускает модель второй раз. Тот же ID с изменённым payload получает HTTP 409. Подробная инструкция, JSON-контракт, схема БД, lifecycle модели и ограничения безопасности: [docs/api_and_postgresql.md](docs/api_and_postgresql.md).

## Блок 8 — Docker quick start

Для запуска всей системы нужен Docker Desktop с Docker Compose. Скопируйте безопасный шаблон окружения и замените пароль:

~~~powershell
Copy-Item .env.docker.example .env.docker
# Задайте свой POSTGRES_PASSWORD в .env.docker
docker compose --env-file .env.docker build
docker compose --env-file .env.docker up -d
docker compose --env-file .env.docker ps
~~~

По умолчанию Swagger доступен на http://127.0.0.1:8000/docs, а MLflow UI — на http://127.0.0.1:5000. Если эти порты заняты локальными процессами, задайте в `.env.docker`, например, `API_HOST_PORT=18000` и `MLFLOW_HOST_PORT=15000`.

Полная Linux-проверка и интеграционный smoke-test запускаются так:

~~~powershell
docker compose --env-file .env.docker --profile test run --rm unit-tests
docker compose --env-file .env.docker --profile test run --rm integration-tests
~~~

Обычная остановка `docker compose --env-file .env.docker down` сохраняет PostgreSQL volume и MLflow-файлы. Команда `down -v` удаляет named volume PostgreSQL вместе с историей прогнозов — используйте её только если действительно хотите стереть контейнерную БД.

Подробное объяснение архитектуры, команд, persistence и диагностики: [docs/docker.md](docs/docker.md).
