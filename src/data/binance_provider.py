"""Binance Spot public REST API implementation."""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Callable

import httpx
import pandas as pd

from .models import MARKET_COLUMNS
from .provider import MarketDataProvider

LOGGER = logging.getLogger(__name__)


class MarketDataDownloadError(RuntimeError):
    """Raised after a market data request cannot be completed."""


class BinanceMarketDataProvider(MarketDataProvider):
    """Download paginated klines from Binance and normalize them."""

    URL = "https://api.binance.com/api/v3/klines"
    LIMIT = 1000

    def __init__(
        self,
        timeout_seconds: float = 15,
        max_retries: int = 3,
        retry_delay_seconds: float = 2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.sleep = sleep
        self.now = now
        self.request_count = 0

    def _request(self, params: dict[str, Any]) -> list[list[Any]]:
        last_error: Exception | None = None
        attempts_made = 0
        for attempt in range(self.max_retries + 1):
            try:
                attempts_made += 1
                self.request_count += 1
                response = self.client.get(self.URL, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable HTTP {response.status_code}", request=response.request, response=response
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise MarketDataDownloadError("Binance returned a non-list response")
                return payload
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in {429} or exc.response.status_code >= 500
                if not retryable or attempt >= self.max_retries:
                    break
                delay = self.retry_delay_seconds * (2 ** attempt)
                LOGGER.warning(
                    "Binance request symbol=%s interval=%s failed; "
                    "retry %d/%d in %.2f seconds: %s",
                    params.get("symbol"),
                    params.get("interval"),
                    attempt + 1,
                    self.max_retries,
                    delay,
                    exc,
                )
                self.sleep(delay)
        raise MarketDataDownloadError(
            f"Binance request failed after {attempts_made} attempts: {last_error}"
        ) from last_error

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None
    ) -> pd.DataFrame:
        cursor = int(start.timestamp() * 1000)
        now_ms = int(self.now().astimezone(timezone.utc).timestamp() * 1000)
        requested_end_ms = int(end.timestamp() * 1000) if end else now_ms
        end_ms = min(requested_end_ms, now_ms)
        rows: list[list[Any]] = []
        while cursor < end_ms:
            page = self._request({"symbol": symbol, "interval": timeframe, "startTime": cursor, "endTime": end_ms, "limit": self.LIMIT})
            if not page:
                break
            if any(len(row) < 7 for row in page):
                raise MarketDataDownloadError("Binance returned a malformed kline")
            eligible = [
                row
                for row in page
                if int(row[0]) < end_ms and int(row[6]) < end_ms
            ]
            rows.extend(eligible)
            LOGGER.info(
                "Binance page symbol=%s interval=%s received=%d closed=%d",
                symbol,
                timeframe,
                len(page),
                len(eligible),
            )
            next_cursor = int(page[-1][0]) + 1
            if next_cursor <= cursor:
                raise MarketDataDownloadError("Binance pagination did not advance")
            cursor = next_cursor
            if len(page) < self.LIMIT:
                break
        if not rows:
            return pd.DataFrame(columns=MARKET_COLUMNS)
        frame = pd.DataFrame(rows)
        normalized = pd.DataFrame({
            "timestamp": pd.to_datetime(frame[0], unit="ms", utc=True),
            "open": pd.to_numeric(frame[1], errors="coerce"),
            "high": pd.to_numeric(frame[2], errors="coerce"),
            "low": pd.to_numeric(frame[3], errors="coerce"),
            "close": pd.to_numeric(frame[4], errors="coerce"),
            "volume": pd.to_numeric(frame[5], errors="coerce"),
        })
        return normalized.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
