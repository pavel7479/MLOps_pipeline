from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest

from src.data.binance_provider import BinanceMarketDataProvider, MarketDataDownloadError
def provider(handler, retries=0):
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=1)
    return BinanceMarketDataProvider(1, retries, 0, client=client, sleep=lambda _: None)


def kline(hour: int) -> list[object]:
    stamp = int(datetime(2024, 1, 1, hour, tzinfo=timezone.utc).timestamp() * 1000)
    return [stamp, "10", "12", "9", "11", "5", stamp + 1, "0", 1, "0", "0", "0"]


def test_normal_response() -> None:
    item = kline(0)
    p = provider(lambda request: httpx.Response(200, json=[item], request=request))
    data = p.fetch_ohlcv("BTCUSDT", "1h", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert len(data) == 1
    assert list(data.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_pagination_and_boundary_deduplication() -> None:
    calls = 0
    first = [kline(0)] * 1000
    first[-1] = kline(1)
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=first if calls == 1 else [kline(1), kline(2)], request=request)
    data = provider(handler).fetch_ohlcv("BTCUSDT", "1h", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert len(data) == 3
    assert calls == 2


def test_empty_response() -> None:
    p = provider(lambda request: httpx.Response(200, json=[], request=request))
    assert p.fetch_ohlcv("BTCUSDT", "1h", datetime.now(timezone.utc) - timedelta(days=1), None).empty


@pytest.mark.parametrize("status", [429, 500])
def test_retries_retryable_http_status(status: int) -> None:
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status if calls == 1 else 200, json=[] if calls > 1 else {}, request=request)
    provider(handler, retries=1).fetch_ohlcv("BTCUSDT", "1h", datetime.now(timezone.utc) - timedelta(days=1), None)
    assert calls == 2


def test_timeout_exhaustion_raises_clear_error() -> None:
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    with pytest.raises(MarketDataDownloadError, match="2 attempts"):
        provider(handler, retries=1).fetch_ohlcv("BTCUSDT", "1h", datetime.now(timezone.utc) - timedelta(days=1), None)


def test_unknown_symbol_400_is_not_retried() -> None:
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"code": -1121}, request=request)
    with pytest.raises(MarketDataDownloadError, match="1 attempts"):
        provider(handler, retries=3).fetch_ohlcv(
            "NOTAREALPAIR", "1h",
            datetime.now(timezone.utc) - timedelta(days=1), None
        )
    assert calls == 1


def test_request_forwards_symbol_interval_and_boundaries() -> None:
    captured = {}
    def handler(request):
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=[], request=request)
    p = provider(handler)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    p.fetch_ohlcv("SOLUSDT", "1d", start, end)
    assert captured["symbol"] == "SOLUSDT"
    assert captured["interval"] == "1d"
    assert int(captured["startTime"]) == int(start.timestamp() * 1000)
    assert int(captured["endTime"]) == int(end.timestamp() * 1000)


def test_retry_delay_uses_exponential_backoff() -> None:
    calls = 0
    delays = []
    def handler(request):
        nonlocal calls
        calls += 1
        status = 500 if calls < 3 else 200
        return httpx.Response(status, json=[] if status == 200 else {}, request=request)
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=1)
    p = BinanceMarketDataProvider(
        1, 2, 2, client=client, sleep=delays.append
    )
    p.fetch_ohlcv(
        "BTCUSDT", "1h",
        datetime.now(timezone.utc) - timedelta(days=1), None
    )
    assert delays == [2, 4]


@pytest.mark.parametrize(
    ("interval", "closed_open", "active_open"),
    [
        ("1h", datetime(2024, 1, 2, 11, tzinfo=timezone.utc), datetime(2024, 1, 2, 12, tzinfo=timezone.utc)),
        ("1d", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc)),
    ],
)
def test_unclosed_candle_is_excluded(interval, closed_open, active_open) -> None:
    now = datetime(2024, 1, 2, 12, 30, tzinfo=timezone.utc)
    closed = kline(0)
    active = kline(0)
    closed[0] = int(closed_open.timestamp() * 1000)
    active[0] = int(active_open.timestamp() * 1000)
    duration = timedelta(hours=1) if interval == "1h" else timedelta(days=1)
    closed[6] = int((closed_open + duration).timestamp() * 1000) - 1
    active[6] = int((active_open + duration).timestamp() * 1000) - 1
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=[closed, active], request=request)
        ),
        timeout=1,
    )
    p = BinanceMarketDataProvider(
        1, 0, 0, client=client, sleep=lambda _: None, now=lambda: now
    )
    frame = p.fetch_ohlcv(
        "BTCUSDT", interval, datetime(2024, 1, 1, tzinfo=timezone.utc), None
    )
    assert frame["timestamp"].tolist() == [pd.Timestamp(closed_open)]
