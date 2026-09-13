from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

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


@pytest.mark.parametrize(
    ("timeframe", "timestamps"),
    [
        ("1h", ["2024-01-02T11:00:00Z", "2024-01-02T12:00:00Z"]),
        ("1d", ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]),
    ],
)
def test_pipeline_persists_only_closed_candles(
    tmp_path: Path, timeframe: str, timestamps: list[str]
) -> None:
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "open": [10.0, 11.0],
        "high": [12.0, 13.0],
        "low": [9.0, 10.0],
        "close": [11.0, 12.0],
        "volume": [1.0, 2.0],
    })
    now = datetime(2024, 1, 2, 12, 30, tzinfo=timezone.utc)
    settings = AppSettings(
        MarketDataSettings(
            "binance", "BTCUSDT", timeframe,
            datetime(2024, 1, 1, tzinfo=timezone.utc), None
        ),
        StorageSettings(tmp_path / "raw", tmp_path / "processed", "parquet"),
        HttpSettings(1, 0, 0), LoggingSettings("INFO"),
    )
    report = MarketDataPipeline(
        settings, FakeProvider(frame), now=lambda: now
    ).run()
    saved = pd.read_parquet(report.processed_path)
    assert len(saved) == 1
    assert saved.iloc[0]["timestamp"] == pd.Timestamp(timestamps[0])
