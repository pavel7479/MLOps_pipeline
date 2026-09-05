from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config.settings import AppSettings, HttpSettings, LoggingSettings, MarketDataSettings, StorageSettings
from src.data.pipeline import MarketDataPipeline
from src.data.provider import MarketDataProvider


class FakeProvider(MarketDataProvider):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def fetch_ohlcv(self, symbol, timeframe, start, end):
        return self.frame.copy()


def test_pipeline_full_chain(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    dirty = pd.concat([valid_frame, valid_frame.iloc[[0]]], ignore_index=True)
    settings = AppSettings(
        MarketDataSettings("binance", "BTCUSDT", "1h", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc)),
        StorageSettings(tmp_path / "raw", tmp_path / "processed", "parquet"),
        HttpSettings(1, 0, 0), LoggingSettings("INFO"),
    )
    report = MarketDataPipeline(settings, FakeProvider(dirty)).run()
    assert report.downloaded_rows == 4
    assert report.final_rows == 3
    assert report.raw_path.exists() and report.processed_path.exists()
    result = pd.read_parquet(report.processed_path)
    assert result["timestamp"].is_unique
