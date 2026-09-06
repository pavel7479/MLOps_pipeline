# Crypto ML Platform — market data ingestion

Учебный production-oriented ML-проект по прогнозированию движения криптовалютного рынка. На текущем этапе реализован только надёжный блок получения, проверки, очистки и хранения исторических OHLCV-данных.

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
