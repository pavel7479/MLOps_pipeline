"""Orchestration for download, validation, cleaning, and persistence."""

from datetime import datetime, timezone
import logging

import pandas as pd

from src.config.settings import AppSettings

from .binance_provider import BinanceMarketDataProvider
from .cleaner import MarketDataCleaner
from .models import DataQualityReport
from .provider import MarketDataProvider
from .storage import ParquetStorage
from .validator import MarketDataValidator

LOGGER = logging.getLogger(__name__)


class MarketDataPipeline:
    """Run the exchange-independent market data ingestion workflow."""

    def __init__(self, settings: AppSettings, provider: MarketDataProvider | None = None) -> None:
        self.settings = settings
        self.provider = provider or self._build_provider()
        self.validator = MarketDataValidator()
        self.cleaner = MarketDataCleaner()
        self.storage = ParquetStorage()

    def _build_provider(self) -> MarketDataProvider:
        if self.settings.market_data.provider != "binance":
            raise ValueError(f"Unsupported provider: {self.settings.market_data.provider}")
        http = self.settings.http
        return BinanceMarketDataProvider(http.timeout_seconds, http.max_retries, http.retry_delay_seconds)

    def run(self) -> DataQualityReport:
        market = self.settings.market_data
        effective_end = market.end_date or datetime.now(timezone.utc)
        LOGGER.info("Starting pipeline provider=%s symbol=%s timeframe=%s range=%s..%s", market.provider, market.symbol, market.timeframe, market.start_date, effective_end)
        raw = self.provider.fetch_ohlcv(market.symbol, market.timeframe, market.start_date, effective_end)
        if raw.empty:
            raise ValueError("Provider returned no market data for the requested range")
        downloaded_rows = len(raw)
        request_count = getattr(self.provider, "request_count", None)
        if request_count is not None:
            LOGGER.info("API requests: %d; downloaded rows: %d", request_count, downloaded_rows)
        date_range = f"{market.start_date.date()}_{effective_end.date()}"
        raw_path = self.settings.storage.raw_dir / f"{market.symbol}_{market.timeframe}_{date_range}.parquet"
        processed_path = self.settings.storage.processed_dir / f"{market.symbol}_{market.timeframe}.parquet"
        if raw_path.exists():
            existing = self.storage.load(raw_path)
            raw = pd.concat([existing, raw], ignore_index=True).drop_duplicates("timestamp", keep="last")
            raw = raw.sort_values("timestamp").reset_index(drop=True)
        self.storage.save(raw, raw_path)
        before = self.validator.validate(raw, market.timeframe)
        LOGGER.info("Raw validation: %s", before)
        cleaned = self.cleaner.clean(raw)
        after = self.validator.validate(cleaned, market.timeframe)
        fatal_errors = tuple(error for error in after.errors if not error.startswith("Missing intervals:"))
        if fatal_errors:
            raise ValueError(f"Cleaned dataset failed validation: {'; '.join(fatal_errors)}")
        self.storage.save(cleaned, processed_path)
        report = DataQualityReport(downloaded_rows, before.duplicate_timestamps, before.invalid_rows, after.missing_intervals, len(cleaned), cleaned["timestamp"].min().to_pydatetime(), cleaned["timestamp"].max().to_pydatetime(), raw_path, processed_path)
        LOGGER.info("Pipeline complete: %s", report)
        return report
