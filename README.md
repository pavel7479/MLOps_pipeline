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
