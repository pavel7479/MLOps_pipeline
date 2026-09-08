# CI/CD и GitHub Container Registry

Блок 9 автоматически проверяет каждое изменение на чистом Linux runner и
публикует runtime-образ API только для успешно проверенного commit ветки
`main`. Это Continuous Delivery: готовый образ появляется в GHCR, но на
production-сервер автоматически не разворачивается.

## Как устроен pipeline

`.github/workflows/ci.yml` запускается на любой `push`, любой
`pull_request`, вручную через `workflow_dispatch` и как reusable workflow
через `workflow_call`. Повторный push в ту же ветку отменяет устаревший CI run.
Каждый GitHub-hosted runner является новой временной Ubuntu-машиной, поэтому
зависимости устанавливаются заново, а pip и Buildx cache только ускоряют сборку.

Обязательная цепочка имеет стабильные имена checks:

1. **Python tests** — Python 3.12, `requirements-test.txt`, `pip check`,
   полный `pytest -v`.
2. **Project checks** — whitespace, компиляция Python и
   `docker compose config`.
3. **Docker runtime build** — multi-stage target `runtime`, Buildx и GHA
   cache; проверяется non-root user, отсутствие pytest, `.env`, `.venv` и
   `BTCUSDT_1h_test.parquet`.
4. **Linux container tests** — target `test`, `pip check` и полный pytest
   внутри Linux image.
5. **Compose integration** — изолированные PostgreSQL, MLflow, Alembic и API;
   ожидание основано на реальных health/state, а не на фиксированных sleep.
   Smoke проверяет prediction, чтение записи из БД, idempotent replay и
   конфликт HTTP 409.

`needs` делает порядок строгим: следующая стадия не начинается после падения
предыдущей. При ошибке Compose печатает `ps` и последние логи, а cleanup
выполняется с `if: always()`.

## Изолированный MLflow для CI

CI не скачивает рыночные данные, не читает Train/Validation/Test datasets, не
запускает обучение или tuning и не меняет настоящий alias `champion`.
`tests/fixtures/feature_manifest.json` и
`tests/fixtures/inference_sample.json` содержат маленький версионированный
контракт из 28 признаков. Скрипт `scripts/create_ci_bootstrap_artifacts.py`
создаёт два детерминированных test-only classifier без вызова `fit`.

`compose.ci.yaml` подключает отдельные CI volumes и профиль
`REGISTRY_BOOTSTRAP_PROFILE=ci-fixture`. Затем обычный Registry bootstrap
регистрирует fixture и API загружает его по настоящему production-пути
`models:/crypto_direction_classifier@champion`. Так проверяется интеграция,
но CI не принимает решений о продвижении ML-моделей.

Файл `.env.ci.example` содержит только публичные одноразовые значения для
временной CI-базы. Пароль `ci_password_disposable_only` не является
production credential и не должен переиспользоваться вне disposable stack.

## Публикация в GHCR

`.github/workflows/publish.yml` запускается:

- автоматически после успешного workflow **CI** для push в `main`;
- вручную из GitHub UI, но только на `main` и только после повторного полного
  reusable CI.

Автоматический gate дополнительно проверяет conclusion, исходное событие
`push`, branch, repository и точный tested commit. Поэтому PR, fork,
отменённый или красный CI не могут публиковать image.

URI вычисляется без захардкоженного пользователя:

~~~text
ghcr.io/<owner>/<repository>/api
~~~

Публикуются два тега:

- `sha-<полный-commit-sha>` — неизменяемая версия для traceability и rollback;
- `main` — удобный указатель на последний зелёный commit основной ветки.

Тег `latest` намеренно не создаётся. Точная идентификация выполняется digest
`sha256:...`: разные теги могут указывать на один digest, но digest однозначно
задаёт содержимое image. После push workflow сам делает `docker pull` по
digest и сверяет OCI labels `org.opencontainers.image.source` и
`org.opencontainers.image.revision`; `created` формирует metadata-action.

Для GHCR используется только встроенный `GITHUB_TOKEN`. PAT и отдельный
пароль registry не нужны. Общие permissions обоих workflows —
`contents: read`; только job **Publish tested API image** дополнительно
получает `packages: write`. Production credentials, SSH secrets и secrets в
Docker build args отсутствуют.

## Закреплённые Actions

Для защиты от неожиданной подмены dependency все actions закреплены на полном
40-символьном commit SHA, а рядом оставлена читаемая версия:

| Action | Версия | Commit SHA |
| --- | --- | --- |
| `actions/checkout` | v7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| `actions/setup-python` | v5.6.0 | `a26af69be951a213d495a4c3e4e4022e16d87065` |
| `docker/setup-buildx-action` | v4.3.0 | `37fe631027851001ddb9b187196cc803df7f5f0e` |
| `docker/login-action` | v4.6.0 | `dbcb813823bdd20940b903addbd779551569679f` |
| `docker/metadata-action` | v6.2.0 | `dc802804100637a589fabce1cb79ff13a1411302` |
| `docker/build-push-action` | v7.2.0 | `f9f3042f7e2789586610d6e8b85c8f03e5195baf` |

## Локальная проверка перед push

Команды выполняются из корня проекта:

~~~powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
docker compose --env-file .env.ci.example -f compose.yaml -f compose.ci.yaml config
docker build --target runtime -t crypto-ml-api:ci .
docker build --target test -t crypto-ml-api-test:ci .
docker run --rm crypto-ml-api-test:ci python -m pip check
docker run --rm crypto-ml-api-test:ci python -m pytest -v
~~~

Полный disposable Compose acceptance использует `.ci-runtime` с fixture
models и отдельное имя проекта. Его точная автоматизированная последовательность
зафиксирована в job **Compose integration** workflow `ci.yml`.

## Обязательная защита ветки main

Workflow-файлы сами не меняют настройки GitHub repository. После первого
успешного CI владелец репозитория должен открыть:

~~~text
Repository → Settings → Rules → Rulesets → New ruleset → New branch ruleset
~~~

Далее:

1. задать имя, например `Protect main`, и статус **Active**;
2. в **Target branches** выбрать `main`;
3. включить **Require a pull request before merging**;
4. включить **Require status checks to pass** и, при наличии в UI,
   **Require branches to be up to date before merging**;
5. выбрать точные checks: **Python tests**, **Project checks**,
   **Docker runtime build**, **Linux container tests**,
   **Compose integration**;
6. запретить bypass, если организационные правила не требуют исключений;
7. сохранить ruleset.

Checks появляются в списке только после того, как соответствующий workflow хотя
бы один раз отчитался GitHub. Настройка считается включённой только после
проверки активного ruleset в UI; наличие этой инструкции само по себе protection
не включает.

## Будущий production deployment и rollback

Реальный deployment в блоке 9 не реализован: не определены сервер, домен,
доступ и HTTPS. Когда target появится, безопасная цепочка будет такой:

~~~text
GHCR image по точному digest
  → GitHub Environment production
  → required reviewer / approval
  → сервер делает pull точного digest
  → docker compose up
  → readiness check
  → deployment success
~~~

В GitHub Environment `production` нужно будет ограничить deployment веткой
`main`, добавить required reviewer, если тариф и тип репозитория это
поддерживают, и хранить только реальные deployment secrets. Эти настройки не
созданы данным блоком и не должны считаться действующими заранее.

Rollback: выбрать digest предыдущего успешного `sha-<commit>`, вернуть его в
deployment-конфигурацию, заново выполнить pull/up и readiness check. Неизменяемые
SHA tags позволяют точно связать Git commit, workflow run и Docker image.

В блок 9 не входят remote deployment, Kubernetes, Terraform, Ansible, выбор
cloud provider, production monitoring, real trading, model retraining и
автоматическое продвижение MLflow alias.
