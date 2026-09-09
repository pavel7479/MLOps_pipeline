# Monitoring runbook

Команды выполняются из корня проекта. Сначала смотрите состояние и последние логи:

```powershell
docker compose --env-file .env.docker ps --all
docker compose --env-file .env.docker logs --tail 200 api monitoring-worker prometheus grafana
```

## API down

Проверьте `/api/v1/health/live`, `/api/v1/health/ready`, затем логи `api`, `postgres`, `mlflow`, `migrate`, `mlflow-init`. Не перезапускайте всю систему без причины. После устранения зависимости выполните `docker compose --env-file .env.docker up -d api` и убедитесь, что `CryptoApiDown` вернулся в inactive.

## PostgreSQL unavailable

`crypto_database_reachable=0` означает, что worker не смог прочитать bounded window. Проверьте `docker compose ... exec postgres pg_isready -U <user> -d <db>`, свободное место и логи. Worker должен остаться running и сам показать 0→1 после восстановления. API readiness может быть unavailable; не отправляйте новые inference-запросы, пока запись результатов не восстановлена.

## MLflow unavailable

Проверьте `http://127.0.0.1:5000/health` и логи MLflow. Worker продолжает PSI и выставляет `crypto_mlflow_reachable=0`. Уже загрузившийся API продолжает использовать модель в памяти, но новый API container не сможет загрузить champion. Восстановите MLflow до restart API.

## Высокая задержка или 5xx

Разделите HTTP p95 и `model.predict` p95. Высокий HTTP при нормальном inference указывает на БД/serialization/network; высокий inference — на модель/CPU. Сравните route и status labels, ресурсы контейнера и database write failures. Порог p95 0.5 s и 5xx 5% — эвристики с минимумом 20 запросов за 5 минут, а не SLA.

## Critical drift

Проверьте размер окна и `crypto_feature_drift_available=1`, затем top-5 PSI и исходные features сохранённых прогнозов. Drift означает отличие от Train, но не доказывает падение F1 или прибыльности. Не запускайте автоматическое retraining и не продвигайте модель автоматически. Сначала расследуйте источник данных, causal timestamp и качество признаков; затем выполните обычный offline Train/Validation/backtest/Registry review.

## Доминирование класса

Rule требует минимум 100 записей и долю одного BUY/HOLD/SELL выше 90% несколько циклов. Проверьте входные распределения, loaded model version и drift. Такое распределение может быть реальным режимом рынка или дефектом данных; одного alert недостаточно для замены модели.

## Rollback приложения

Найдите предыдущий проверенный GHCR digest/`sha-<commit>`, задайте его для API image и пересоздайте `api` и `monitoring-worker`. Они используют один runtime image. Не используйте изменяемый `latest`. Убедитесь в health, `/metrics`, target UP и сохранности prediction history.

## Rollback модели

Верните alias `champion` на ранее одобренную MLflow model version, затем контролируемо перезапустите только API. Проверьте `/api/v1/model` и `crypto_loaded_model_info`. Старые prediction rows сохраняют свою model version; replay не пересчитывает их новой моделью.

## Остановка и сохранность

```powershell
docker compose --env-file .env.docker down
docker compose --env-file .env.docker up -d
```

Обычный `down` сохраняет PostgreSQL predictions, MLflow Registry/artifacts, Prometheus history и Grafana state. Команда `down -v` удаляет named volumes PostgreSQL, Prometheus и Grafana; используйте её только при осознанном полном сбросе локальных данных.
