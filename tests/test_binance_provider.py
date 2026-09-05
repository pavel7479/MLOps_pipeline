from datetime import datetime, timedelta, timezone

import httpx
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
