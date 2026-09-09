# Архитектура проекта

## Поток данных и модели

```text
Binance OHLCV -> очистка -> causal features/target -> chronological Train/Validation/Test
                                      |
                                      +-> обучение и time-series CV
                                      +-> Validation backtest
                                      +-> MLflow Registry: champion/challenger
```

Признаки строки `t` рассчитываются по полностью закрытой свече `t`; inference разрешён после её закрытия. Target — изменение Close относительно `close[t]` через `target.horizon_hours`. Test не используется при выборе или мониторинге модели.

## Runtime

```text
                         +---------------- MLflow ----------------+
                         | tracking + Registry + model artifacts  |
                         +-------------------+---------------------+
                                             |
PostgreSQL <- Alembic <- FastAPI API <--------+     /metrics :8000
    |                    |
    | last 500 rows      +-- POST /predict -> committed prediction
    v
monitoring-worker :9101 -- Train-only PSI + DB/MLflow reachability
    |                                  |
    +--------------+-------------------+
                   v
             Prometheus :9090 -> rules + 7-day history
                   |
                   v
              Grafana :3000 -> two provisioned dashboards
```

API не зависит от worker, Prometheus или Grafana: отказ monitoring plane не блокирует inference. Worker зависит только от готовой PostgreSQL schema; MLflow проверяется независимо и его отказ не завершает worker. PostgreSQL и worker не публикуют host ports.

Persistent state разделён: PostgreSQL — named volume `crypto-ml-platform_postgres_data`; Prometheus — `crypto-ml-platform_prometheus_data`; Grafana — `crypto-ml-platform_grafana_data`; MLflow — bind mounts `data/mlflow` и `mlartifacts`. Точное имя получает префикс Compose project name, если он переопределён.

## Три независимых жизненных цикла

1. **CI/CD приложения:** tests -> runtime image -> Linux tests -> Compose integration -> GHCR image по immutable `sha-<commit>`.
2. **Жизненный цикл модели:** обучение -> Validation -> backtest gate -> MLflow model version -> alias `champion`/`challenger`.
3. **Жизненный цикл приложения:** Docker image version/digest -> контейнеры API и worker -> rollback image.

Версия MLflow model — не версия Docker image. Смена alias модели требует контролируемого restart API, чтобы он один раз загрузил новую версию. Rollback приложения и rollback модели выполняются разными процедурами.

CI использует 28 versioned fixture-признаков и маленькие заранее созданные модели, не скачивает Binance data, не обучает модель, не читает Train/Validation/Test parquet и не меняет настоящий Registry.
