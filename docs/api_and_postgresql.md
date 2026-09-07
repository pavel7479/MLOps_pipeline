# FastAPI inference API и PostgreSQL

## Назначение

Блок 7 превращает зарегистрированную ML-модель в локальный HTTP-сервис. Клиент передаёт готовые признаки, сервис проверяет контракт, получает класс `BUY`, `HOLD` или `SELL` и сохраняет неизменяемый факт прогноза в PostgreSQL.

Это inference API, а не торговый робот. Оно не выставляет ордера, не даёт финансовых рекомендаций, не загружает свечи с Binance и не рассчитывает признаки. Рыночные данные и feature engineering остаются отдельными предыдущими этапами pipeline.

## Архитектура и lifecycle

При старте приложения FastAPI:

1. создаётся один SQLAlchemy Engine и фабрика request-scoped сессий;
2. проверяется доступность PostgreSQL;
3. `MLflowModelProvider` разрешает alias `champion` модели `crypto_direction_classifier`;
4. соответствующая MLflow PyFunc-модель загружается в память ровно один раз;
5. из MLflow signature или artifact `feature_manifest.json` читается порядок признаков.

На каждом HTTP-запросе модель заново не загружается. Сессия SQLAlchemy создаётся на запрос, а после ответа закрывается. При остановке приложения принадлежащий ему Engine освобождает пул подключений.

Изменение alias `champion` в Registry не меняет уже работающий процесс. Чтобы безопасно подхватить новую версию, перезапустите API. Исторические строки продолжают хранить имя, alias, точный номер версии и run ID той модели, которая реально сформировала ответ.

API зависит только от общего протокола модели с методом `predict`. Модули FastAPI и inference не импортируют CatBoost, XGBoost или LightGBM, поэтому реализацию модели можно заменить через Registry или тестовый `ModelProvider`.

## Конфигурация

Обычные несекретные параметры находятся в `config.yaml`:

~~~yaml
api:
  host: "127.0.0.1"
  port: 8000

inference:
  registered_model_name: "crypto_direction_classifier"
  model_alias: "champion"
  expected_symbol: "BTCUSDT"
  expected_timeframe: "1h"

database:
  url_env_variable: "DATABASE_URL"
  pool_pre_ping: true
~~~

Строка подключения — секрет и задаётся через локальный `.env`:

~~~dotenv
DATABASE_URL=postgresql+psycopg://mlops_user:STRONG_PASSWORD@127.0.0.1:5432/mlops_pipeline
~~~

`.env` и `.env.*` исключены из Git; `.env.example` содержит только заглушку. Используется синхронный драйвер psycopg 3. SQLite в рабочем API намеренно не поддерживается.

## Подготовка PostgreSQL

Установите локальный PostgreSQL 17 или совместимую актуальную версию. Пример SQL от имени администратора:

~~~sql
CREATE ROLE mlops_user LOGIN PASSWORD 'STRONG_PASSWORD';
CREATE DATABASE mlops_pipeline OWNER mlops_user;
~~~

Создание и изменение таблиц выполняет только Alembic:

~~~powershell
..venvScriptsalembic.exe upgrade head
..venvScriptspython.exe scriptscheck_database.py
~~~

Первичная миграция `20260907_0001` создаёт таблицу `predictions`. Для отката одного шага используйте `alembic downgrade -1` только если осознанно хотите удалить таблицу и её данные.

## Запуск

Сначала запустите локальный MLflow:

~~~powershell
..venvScriptsmlflow.exe server `
  --backend-store-uri sqlite:///data/mlflow/mlflow.db `
  --artifacts-destination ./mlartifacts `
  --host 127.0.0.1 `
  --port 5000
~~~

Затем в другом терминале запустите API из корня проекта:

~~~powershell
..venvScriptspython.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
~~~

Откройте:

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- liveness: http://127.0.0.1:8000/api/v1/health/live
- readiness: http://127.0.0.1:8000/api/v1/health/ready

`live` показывает, что HTTP-процесс отвечает. `ready` возвращает 200 только когда модель загружена и PostgreSQL доступен; иначе возвращает 503 с отдельными состояниями `model` и `database`. Ошибка загрузки champion при обычном старте останавливает приложение, чтобы оно не принимало запросы без модели.

## Контракт POST /api/v1/predict

Пример формы запроса:

~~~json
{
  "request_id": "ca88d2e3-1087-4d95-ad3e-b481f4e359e7",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "feature_timestamp": "2025-01-15T12:00:00+00:00",
  "features": {
    "return_1h": 0.001
  }
}
~~~

Объект `features` в настоящем запросе обязан содержать ровно все 28 имён из MLflow manifest. Сокращённый пример выше показывает только форму и сам по себе получит 422.

Правила валидации:

- `request_id` — UUID;
- разрешены только `BTCUSDT` и `1h`, заданные в конфигурации;
- `feature_timestamp` обязан содержать timezone;
- отсутствующие и лишние признаки запрещены;
- каждое значение — настоящий JSON number, но не строка и не boolean;
- `NaN` и бесконечности запрещены;
- перед моделью колонки переставляются в точный порядок manifest.

Timestamp относится к полностью закрытой свече `t`. Прогноз разрешено отправлять только после закрытия этой свечи. Модель оценивает движение Close относительно `close[t]` на горизонте `target.horizon_hours`; API не может самостоятельно доказать, что клиент дождался закрытия свечи, поэтому это часть контракта вызывающей стороны.

Успешный ответ содержит класс, точную ссылку на модель, latency, время сохранения и признак replay:

~~~json
{
  "request_id": "ca88d2e3-1087-4d95-ad3e-b481f4e359e7",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "feature_timestamp": "2025-01-15T12:00:00Z",
  "prediction": "HOLD",
  "model": {
    "registered_name": "crypto_direction_classifier",
    "alias": "champion",
    "version": "1",
    "run_id": "..."
  },
  "latency_ms": 1.23,
  "replayed": false,
  "created_at": "2026-09-07T12:00:00Z"
}
~~~

Коды ошибок: 422 для нарушения входного контракта, 404 для неизвестного `request_id`, 409 для idempotency-конфликта, 503 для недоступной БД и 500 для нарушения контракта моделью или неожиданной внутренней ошибки. Внешнему клиенту не возвращаются stack trace, SQL или секреты.

## Идемпотентность

Перед inference сервис строит детерминированный SHA-256 от канонического payload: symbol, timeframe, timestamp и всех признаков в порядке manifest.

- Новый `request_id`: модель вызывается, результат сохраняется, `replayed=false`.
- Тот же ID и тот же hash: возвращается сохранённая строка, `replayed=true`; модель не вызывается.
- Тот же ID и другой payload: HTTP 409; исходная строка остаётся неизменной.

Проверка выполняется до вызова модели и защищена уникальным ограничением PostgreSQL. Конкурентная гонка при вставке обрабатывается повторным чтением. Replay возвращает исходную версию модели даже если после первого запроса Registry alias уже переключили.

## Схема хранения и история

Таблица `predictions` хранит:

- внутренний UUID и уникальный `request_id`;
- SHA-256 payload;
- время признаков и время создания записи;
- symbol, timeframe и JSONB со всеми признаками;
- класс `BUY/HOLD/SELL` с CHECK constraint;
- имя Registry-модели, alias, version и run ID;
- latency inference.

Индексы ускоряют поиск по времени, классу и версии модели. История доступна через:

- `GET /api/v1/predictions?limit=50&offset=0`;
- `GET /api/v1/predictions?prediction=BUY`;
- `GET /api/v1/predictions/{request_id}`;
- `GET /api/v1/model` для метаданных модели в памяти.

Сырые свечи, target, test dataset и торговые операции в PostgreSQL этого блока не записываются.

## Проверки

Изолированные unit/API-тесты используют fake model providers и отдельную SQLite БД только внутри тестов:

~~~powershell
..venvScriptspython.exe -m pytest -v
~~~

Реальная проверка требует работающих MLflow, PostgreSQL и API. Скрипт берёт одну строку Validation, проверяет readiness/model, создаёт prediction, читает его из PostgreSQL, повторяет запрос и проверяет конфликт:

~~~powershell
..venvScriptspython.exe scriptssmoke_test_api.py
~~~

## Ограничения безопасности

Текущая конфигурация рассчитана только на локальную разработку: bind `127.0.0.1`, без authentication, authorization, HTTPS и rate limiting. Не публикуйте порт 8000 или PostgreSQL в интернет. Перед production-развёртыванием нужны TLS/reverse proxy, аутентификация, авторизация, управление секретами, сетевые ACL, ограничение размера/частоты запросов, централизованные логи, мониторинг, backup и политика хранения feature payload.
