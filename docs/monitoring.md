# Monitoring, Prometheus, Grafana и ML drift

Блок 10 добавляет локальную наблюдаемость без изменения модели и без автоматической торговли. FastAPI публикует сервисные и inference-метрики на `/metrics`. Отдельный процесс `monitoring-worker` читает последние 500 прогнозов из PostgreSQL, проверяет доступность MLflow и рассчитывает PSI по 28 признакам. Prometheus собирает метрики каждые 15 секунд, а Grafana показывает два готовых dashboard.

## Понятия простыми словами

- **Metric** — числовое наблюдение о системе.
- **Counter** — счётчик, который только растёт, например число новых прогнозов или конфликтов.
- **Gauge** — текущее значение, которое может расти и уменьшаться: доступность БД, размер окна, доля BUY.
- **Histogram** — распределение измерений по диапазонам. Из него считаются квантили задержки.
- **Prometheus** — сервис, который регулярно забирает `/metrics`, хранит временные ряды и вычисляет alert rules.
- **Grafana** — интерфейс графиков поверх данных Prometheus.
- **Dashboard** — сохранённый набор панелей и запросов.
- **Alert** — правило, которое переходит в pending/firing при устойчивом нарушении порога. В этом локальном этапе нет Alertmanager и внешних уведомлений.
- **Drift** — изменение распределения входных признаков относительно Train.
- **PSI** (Population Stability Index) — мера отличия долей наблюдений в одинаковых диапазонах Train и текущего окна.
- **p50/p95/p99** — задержка, быстрее которой завершились соответственно 50%, 95% и 99% запросов. Например, p95 = 0.4 секунды означает: 95 из 100 запросов были не медленнее 0.4 секунды; оставшиеся 5 могли быть медленнее.

Drift не доказывает, что модель стала плохой: рынок мог измениться, но решение всё ещё может быть полезным. Отсутствие drift также не доказывает точность модели. Live F1 нельзя честно посчитать сразу после inference: истинный класс появляется только после закрытия будущей свечи на горизонте `target.horizon_hours` (сейчас 3 часа). Связывание прогноза с будущим outcome — отдельная будущая задача; в Блоке 10 оно не реализовано.

## Метрики API

`crypto_http_requests_total{method,route,status_code}` и `crypto_http_request_duration_seconds{method,route}` используют шаблон маршрута. UUID, timestamp, IP, SQL и request payload не являются labels. Поэтому все запросы вида `/api/v1/predictions/<uuid>` попадают в один ряд с `route="/api/v1/predictions/{request_id}"`.

`crypto_ml_inference_duration_seconds{model_version}` измеряет только вызов `model.predict`. `crypto_ml_predictions_total{prediction,model_version}` растёт лишь после успешного commit новой строки. Replay учитывается отдельно в `crypto_ml_prediction_replays_total`; конфликт — в `crypto_ml_prediction_conflicts_total`; ошибка записи — в `crypto_database_write_failures_total`.

`crypto_loaded_model_info{registered_model,alias,version,backtest_status}=1` описывает загруженную модель без high-cardinality `run_id`. `crypto_model_loaded` равен 0 или 1. Endpoint `/metrics` скрыт из Swagger и при запросе не обращается к PostgreSQL.

## Train-only PSI reference

Reference создаётся командой:

```powershell
.\.venv\Scripts\python.exe scripts\build_monitoring_reference.py
```

Скрипт читает только `data/ml/BTCUSDT_1h_train.parquet` и `data/ml/feature_manifest.json`. Validation, Test и общий dataset не используются. В Git сохраняется компактный `monitoring/reference/BTCUSDT_1h_train_reference.json`: SHA256 входов, Train-период, число строк, точный контракт 28 признаков, статистики и до 10 квантильных bins каждого признака. В runtime image входит JSON, но не Parquet.

PSI использует сглаживание epsilon, корректно работает с пустыми ожидаемыми bins, одинаковыми quantile boundaries и значениями за Train-диапазоном: последние попадают в крайние bins. Состояния по умолчанию:

- `stable`: PSI < 0.10;
- `warning`: 0.10 ≤ PSI < 0.25;
- `critical`: PSI ≥ 0.25;
- `insufficient_data`: строк меньше 100, PSI не объявляется доступным.

Эти пороги — операционные эвристики, а не доказательство финансового качества.

## Worker и dashboards

Worker запускается как `python -m src.monitoring.worker`, не публикует порт на host и отдаёт метрики на `monitoring-worker:9101/metrics` внутри Compose network. SQL-запрос выбирает только `features`, `prediction`, `created_at`, содержит `LIMIT 500`. При отказе PostgreSQL worker остаётся жив, выставляет `crypto_database_reachable=0` и повторяет попытку. Недоступность MLflow также не останавливает PSI.

Grafana автоматически получает datasource `http://prometheus:9090` и dashboards:

- `Crypto ML — Service Overview` — статусы, traffic, 5xx, p50/p95/p99, inference p95, прогнозы/replay/conflicts и версия модели;
- `Crypto ML — Model Monitoring` — размер окна, BUY/HOLD/SELL shares, доступность drift, все 28 PSI, top-5 и время последнего расчёта.

Prometheus содержит восемь локальных правил: недоступность API/worker/PostgreSQL/MLflow, высокий 5xx rate, p95 выше 0.5 s, critical drift и доминирование одного класса. Для rate/latency/class rules задан минимальный traffic/window и выдержка во времени, чтобы единичное событие не создавало ложный сигнал.

## Grafana dashboards

Оба dashboard автоматически provision из JSON в monitoring/grafana/dashboards/; ручной импорт и сохранение через UI не требуются. Стабильные UID — crypto-ml-service-overview и crypto-ml-model-monitoring, refresh — 15 секунд, timezone — браузер пользователя.

Crypto ML — Service Overview организован сверху вниз по секциям SYSTEM STATUS, TRAFFIC & ERRORS, LATENCY и PREDICTIONS. Статусы API/worker показываются как UP/DOWN, PostgreSQL и MLflow — как AVAILABLE/UNAVAILABLE; отдельно видны текущая Registry-модель, total/5xx/error rate, request rate, HTTP и inference latency, BUY/HOLD/SELL, replay, conflict и DB write failures.

Crypto ML — Model Monitoring содержит секции MODEL & DATA STATUS, RECENT PREDICTIONS, FEATURE DRIFT и DRIFT HISTORY. DRIFT STATUS = READY означает только достаточный объём данных для PSI. Пока данных меньше минимума, warning/critical и все PSI-панели показывают NOT CALCULATED/NO DATA, а не ложный PSI=0. LAST DRIFT CHECK переводит Unix seconds в миллисекунды для корректного datetime; selector Feature фильтрует PSI history по label из Prometheus.

Если один из классов ещё не встречался, current distribution создаёт нулевую BUY/HOLD/SELL series на уровне PromQL, не изменяя backend metrics. Полные графики истории показывают только фактически опубликованные Prometheus series.

## Границы этапа

Monitoring локальный. Здесь нет Alertmanager, email/Telegram/PagerDuty, Loki/ELK, OpenTelemetry, live Binance ingestion, live ground truth/F1, автоматического retraining/promotion, Kubernetes, cloud deployment и реальной торговли.
