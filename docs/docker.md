# Docker и Docker Compose

Блок 8 упаковывает работающую систему Блоков 1–7 в воспроизводимое локальное Linux-окружение. Модели не переобучаются, признаки и target не меняются, Binance при старте не вызывается, а Test dataset не используется.

## Основные понятия

- **Docker image** — неизменяемый шаблон файловой системы и окружения приложения.
- **Container** — запущенный экземпляр image. Его можно удалить и создать снова из того же образа.
- **Dockerfile** — инструкция, по которой строится image приложения.
- **Docker Compose** — описание нескольких связанных контейнеров, их порядка запуска, портов, сети и хранилищ.
- **Volume** — отдельное постоянное хранилище. Оно переживает удаление и пересоздание контейнера.
- **Network** — внутренняя сеть, где контейнеры находят друг друга по именам сервисов.
- **Healthcheck** — функциональная проверка готовности сервиса. Статус `running` означает только запущенный процесс, а `healthy` — что сервис уже способен принимать работу.
- **Multi-stage build** — сборка из нескольких стадий. Инструменты и wheels готовятся в builder, тесты находятся в test image, а конечный runtime получает только необходимое API.

## Архитектура

~~~text
PostgreSQL ── healthy ──> migrate (Alembic, exit 0) ─┐
                                                     ├─> API ─> healthy
MLflow ───── healthy ──> mlflow-init (exit 0) ──────┘
~~~

Долгоживущие сервисы `postgres`, `mlflow` и `api` имеют policy `unless-stopped`. `migrate` и `mlflow-init` — одноразовые задачи; состояние `Exited (0)` для них правильно.

Все сервисы находятся в одной bridge-сети `mlops_network`. Docker DNS предоставляет имена `postgres`, `mlflow` и `api`. IP-адреса контейнеров не фиксируются.

### Почему внутри контейнера нельзя использовать localhost

`localhost` внутри API-контейнера означает сам API-контейнер, а не Windows host и не соседний контейнер. Поэтому Compose передаёт:

~~~text
DATABASE_HOST=postgres
MLFLOW_HOST=mlflow
DATABASE_URL=postgresql+psycopg://...@postgres:5432/mlops_pipeline
MLFLOW_TRACKING_URI=http://mlflow:5000
~~~

Host-порты нужны только браузеру и инструментам Windows:

~~~text
Windows -> 127.0.0.1:8000 -> api:8000
Windows -> 127.0.0.1:5000 -> mlflow:5000
~~~

PostgreSQL port на Windows не публикуется. Это исключает конфликт с локальной PostgreSQL и уменьшает доступную извне поверхность.

## Что хранится постоянно

`postgres_data` — named volume, смонтированный в `/var/lib/postgresql/data`. История прогнозов живёт отдельно от контейнера PostgreSQL:

~~~text
удалили/пересоздали PostgreSQL container
        ↓
postgres_data остался
        ↓
prediction history сохранилась
~~~

MLflow сохраняет metadata в bind mount `./data/mlflow:/mlflow/data` и run artifacts в `./mlartifacts:/mlflow/artifacts`. Backend URI внутри контейнера — `sqlite:////mlflow/data/mlflow.db`.

Обычная команда `docker compose ... down` не удаляет named volumes и bind-mounted файлы.

> **Осторожно:** `docker compose ... down -v` удаляет `postgres_data` и контейнерную историю прогнозов. Эта команда не нужна для обычной остановки.

## Первый запуск

Требуется Docker Desktop с Compose v2. Команды выполняются из корня проекта:

~~~powershell
Copy-Item .env.docker.example .env.docker
~~~

Откройте `.env.docker` и замените `POSTGRES_PASSWORD` на сильный URL-safe пароль. Файл локальный и исключён из Git. Пароль не передаётся как build argument и не попадает в image.

Если локальные API или MLflow уже занимают стандартные порты, измените только host-порты:

~~~dotenv
API_HOST_PORT=18000
MLFLOW_HOST_PORT=15000
~~~

Внутренние адреса `api:8000`, `mlflow:5000` и `postgres:5432` при этом не меняются.

Проверка конфигурации, сборка и запуск:

~~~powershell
docker compose --env-file .env.docker config
docker compose --env-file .env.docker build
docker compose --env-file .env.docker up -d
docker compose --env-file .env.docker ps
~~~

Ожидаемое состояние:

~~~text
postgres     Up ... (healthy)
mlflow       Up ... (healthy)
api          Up ... (healthy)
migrate      Exited (0)
mlflow-init  Exited (0)
~~~

Порядок обеспечивается условиями `service_healthy` и `service_completed_successfully`. Alembic не запускается внутри startup API. Registry также проверяется до загрузки champion.

## Доступ

При стандартных портах:

- Swagger UI: http://127.0.0.1:8000/docs
- MLflow UI: http://127.0.0.1:5000
- readiness: http://127.0.0.1:8000/api/v1/health/ready
- сведения о модели: http://127.0.0.1:8000/api/v1/model

Если в `.env.docker` указаны другие `API_HOST_PORT` и `MLFLOW_HOST_PORT`, используйте их вместо 8000 и 5000.

## Registry initialization

`mlflow-init` после healthcheck MLflow проверяет модель `crypto_direction_classifier` и aliases `champion`/`challenger`.

- Если оба alias существуют, задача завершает работу без новых versions.
- Если Registry пуст, она импортирует сохранённые baseline и tuned LightGBM artifacts и feature manifest без обучения или tuning.
- Если нужных файлов нет, задача завершается non-zero с понятным перечнем отсутствующих artifacts, и API не стартует.

Bootstrap-файлы доступны read-only только одноразовому `mlflow-init`. API их не монтирует.

## Миграции

`migrate` ждёт healthy PostgreSQL и выполняет:

~~~text
alembic upgrade head
~~~

Команда идемпотентна: повторный запуск не создаёт таблицы или строки заново. Текущий head — `20260907_0001`.

Ручная повторная проверка:

~~~powershell
docker compose --env-file .env.docker run --rm migrate
docker compose --env-file .env.docker exec postgres psql -U mlops_user -d mlops_pipeline -c "SELECT version_num FROM alembic_version;"
~~~

## Логи и диагностика

~~~powershell
docker compose --env-file .env.docker logs -f api
docker compose --env-file .env.docker logs -f mlflow
docker compose --env-file .env.docker logs -f postgres
docker compose --env-file .env.docker logs migrate
docker compose --env-file .env.docker logs mlflow-init
docker compose --env-file .env.docker exec api id
docker compose --env-file .env.docker exec api python -m pip check
~~~

Логи идут в stdout/stderr. Для json-file настроена ротация: максимум 3 файла по 10 MB на контейнер. Секреты и полный `DATABASE_URL` не логируются.

PostgreSQL диагностируется внутри сети, без открытия host-порта:

~~~powershell
docker compose --env-file .env.docker exec postgres pg_isready -U mlops_user -d mlops_pipeline
~~~

## Контейнерные тесты

Test target содержит pytest и тестовые зависимости, которых нет в runtime image. Unit suite не обращается к Binance и не требует работающих внешних сервисов:

~~~powershell
docker compose --env-file .env.docker --profile test run --rm unit-tests
~~~

Integration smoke-test ждёт healthy API, использует Validation sample и проверяет `/live`, `/ready`, `/model`, POST `/predict`, чтение сохранённого прогноза, идемпотентный replay и HTTP 409:

~~~powershell
docker compose --env-file .env.docker --profile test run --rm integration-tests
~~~

Только Validation parquet монтируется read-only в integration-test container. `BTCUSDT_1h_test.parquet` не копируется в image и не монтируется ни в один сервис.

## Изменение кода и остановка

Пересобрать и заменить API после изменения кода:

~~~powershell
docker compose --env-file .env.docker build api
docker compose --env-file .env.docker up -d api
~~~

Перезапустить отдельный сервис:

~~~powershell
docker compose --env-file .env.docker restart api
docker compose --env-file .env.docker restart postgres
~~~

Безопасно остановить систему, сохранив данные:

~~~powershell
docker compose --env-file .env.docker down
~~~

## Безопасность и границы этапа

API работает как `appuser` с UID/GID 10001 и `no-new-privileges`, не является privileged, не получает Docker socket и host source bind mount. `.dockerignore` исключает Git, host `.venv`, secrets, datasets, локальные БД, MLflow artifacts, логи и caches. В runtime образе нет pytest, Train/Validation/Test parquet и Windows paths.

Это локальное Docker-окружение: один API worker, MLflow metadata остаётся в SQLite. Здесь нет cloud deployment, CI/CD, Kubernetes, Nginx, HTTPS, authentication, Redis/Celery/Kafka, monitoring stack и горизонтального масштабирования.
