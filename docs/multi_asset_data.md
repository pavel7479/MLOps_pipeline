# Блок 11: исторические данные по шести криптовалютам

## Что реализовано

Один и тот же pipeline получает публичные OHLCV-свечи Binance для шести пар:
BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT, DOGEUSDT.
Для каждой пары сохраняются независимые наборы 1h и 1d, всего 12 наборов.
Списки находятся в секции market_data файла config.yaml; в Python-коде перечня
монет нет. Чтобы добавить новую пару, достаточно изменить конфигурацию.

Поля symbol: BTCUSDT и timeframe: 1h намеренно сохранены. Их использует старый
одиночный pipeline и ML-контур Блоков 1–10. Новые поля symbols и intervals
используются массовым запуском.

## Команды

Все команды выполняются из корня проекта.

Полная загрузка и проверка 6 × 2:

~~~powershell
.\.venv\Scripts\python.exe scripts\download_market_data.py --all
~~~

Только один набор:

~~~powershell
.\.venv\Scripts\python.exe scripts\download_market_data.py --symbol ETHUSDT --interval 1d
~~~

Старая команда сохранена и по-прежнему загружает legacy-набор BTCUSDT 1h:

~~~powershell
.\.venv\Scripts\python.exe scripts\download_market_data.py
~~~

Небольшой реальный smoke-test Binance (не скачивает многолетнюю историю и
ничего не сохраняет):

~~~powershell
.\.venv\Scripts\python.exe scripts\smoke_test_binance.py
~~~

Linux-команды отличаются только путём к Python:

~~~bash
python scripts/download_market_data.py --all
python scripts/download_market_data.py --symbol BTCUSDT --interval 1h
python scripts/smoke_test_binance.py
~~~

## Где лежат данные

Архитектура Блока 1 сохранена:

- raw-снимок: data/raw/<SYMBOL>_<INTERVAL>_<START>_<END>.parquet;
- проверенный набор: data/processed/<SYMBOL>_<INTERVAL>.parquet;
- manifest: data/processed/market_data_manifest.json.

Например, data/processed/ETHUSDT_1d.parquet содержит только дневные свечи
ETHUSDT. Монеты не смешиваются в одном файле. Parquet и manifest являются
генерируемыми данными и не коммитятся в Git.

## Гарантии данных

Провайдер делает пагинацию Binance с лимитом 1000 строк. Timeout, HTTP 429 и
временные 5xx повторяются ограниченное число раз с экспоненциальной задержкой.
Загрузка выполняется последовательно: надёжность и соблюдение rate limit здесь
важнее небольшой экономии времени.

Каждый набор проходит единый контракт:

- колонки timestamp, open, high, low, close, volume;
- время приведено к UTC и отсортировано по возрастанию;
- timestamp уникален внутри пары symbol + interval;
- цены положительные, volume неотрицательный;
- проверяются OHLC-соотношения;
- не допускаются NaN, None, inf и -inf;
- разрывы шага 1h или 1d считаются и записываются, но не заполняются
  искусственными свечами;
- Parquet сначала пишется во временный файл, затем атомарно заменяет рабочий.

Отсутствие истории до листинга монеты не считается ошибкой: actual_start в
manifest показывает первую реально доступную свечу. Монеты не обрезаются до
общей даты начала.

В набор попадают только полностью закрытые свечи. Провайдер проверяет Binance
close time, а pipeline повторно проверяет timestamp + interval <= текущее UTC
время. Поэтому текущий незакрытый час и текущий незакрытый день не становятся
историческим фактом.

Повторный запуск объединяет raw-данные по timestamp, сортирует их и атомарно
обновляет тот же processed-файл. Дубликаты не накапливаются.

## Manifest и итоговый статус

Для каждого файла manifest хранит symbol, interval, число строк, первую и
последнюю свечу, requested/actual start, число дубликатов и пропусков, статус
валидации, путь, SHA-256, время создания и provider. SHA-256 вычисляется по
байтам уже сохранённого Parquet-файла.

Посмотреть сводку:

~~~powershell
$manifest = Get-Content data\processed\market_data_manifest.json -Raw |
  ConvertFrom-Json
$manifest.last_run
$manifest.datasets |
  Format-Table symbol, interval, row_count, first_timestamp, last_timestamp,
    missing_intervals, validation_status
~~~

Сверить хеш конкретного файла с manifest:

~~~powershell
Get-FileHash data\processed\BTCUSDT_1h.parquet -Algorithm SHA256
~~~

Если одна пара не загрузилась, остальные пары всё равно обрабатываются. Ошибка
попадает в last_run.failures, общий статус становится FAIL, а CLI завершается
ненулевым кодом. Поэтому частичный результат невозможно принять за полный успех.

## Граница Блока 11

1h подготовлен для будущей торговой ML-модели. 1d подготовлен для будущих
признаков дневного тренда, EMA и волатильности в Блоке 12. В Блоке 11 не
рассчитываются признаки, EMA/RSI/MACD/ATR, target и сигналы, не переобучаются
модели и не изменяются MLflow, FastAPI, PostgreSQL, Prometheus или Grafana.

Большие исторические файлы исключены из Docker build context. Test image
содержит код и PyArrow, поэтому полный fake-provider сценарий 6 × 2 выполняется
в Linux-контейнере без зависимости от интернета. Реальный Binance smoke
запускается отдельно.
