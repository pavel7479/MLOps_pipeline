from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from src.config.settings import (
    AppSettings,
    HttpSettings,
    LoggingSettings,
    MarketDataSettings,
    StorageSettings,
)
from src.data.multi_asset import MultiAssetMarketDataPipeline
from src.data.provider import MarketDataProvider


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT")
INTERVALS = ("1h", "1d")
NOW = datetime(2024, 1, 5, 12, 30, tzinfo=timezone.utc)


class MatrixProvider(MarketDataProvider):
    def __init__(self, fail_pair=None) -> None:
        self.calls = []
        self.fail_pair = fail_pair

    def fetch_ohlcv(self, symbol, timeframe, start, end):
        self.calls.append((symbol, timeframe))
        if (symbol, timeframe) == self.fail_pair:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        frequency = "h" if timeframe == "1h" else "D"
        timestamps = pd.date_range("2024-01-01", periods=3, freq=frequency, tz="UTC")
        offset = float(SYMBOLS.index(symbol))
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": [10 + offset, 11 + offset, 12 + offset],
            "high": [12 + offset, 13 + offset, 14 + offset],
            "low": [9 + offset, 10 + offset, 11 + offset],
            "close": [11 + offset, 12 + offset, 13 + offset],
            "volume": [1.0, 2.0, 3.0],
        })


def settings_for(tmp_path: Path) -> AppSettings:
    return AppSettings(
        MarketDataSettings(
            "binance",
            "BTCUSDT",
            "1h",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            None,
            SYMBOLS,
            INTERVALS,
        ),
        StorageSettings(tmp_path / "raw", tmp_path / "processed", "parquet"),
        HttpSettings(1, 0, 0),
        LoggingSettings("INFO"),
    )


def test_fake_provider_builds_all_twelve_datasets_and_manifest(tmp_path: Path) -> None:
    provider = MatrixProvider()
    pipeline = MultiAssetMarketDataPipeline(
        settings_for(tmp_path), provider=provider, now=lambda: NOW
    )
    result = pipeline.run()
    assert result.is_success
    assert result.successful_datasets == result.requested_datasets == 12
    assert set(provider.calls) == {
        (symbol, interval) for symbol in SYMBOLS for interval in INTERVALS
    }
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["last_run"]["status"] == "PASS"
    assert len(payload["datasets"]) == 12
    for item in payload["datasets"]:
        assert item["row_count"] == 3
        assert item["duplicate_timestamps"] == 0
        assert item["validation_status"] == "PASS"
        assert item["actual_start"] == item["first_timestamp"]
        assert len(item["file_hash"]) == 64
        assert Path(item["file_path"]).exists()


def test_rerun_is_idempotent_and_preserves_unique_timestamps(tmp_path: Path) -> None:
    pipeline = MultiAssetMarketDataPipeline(
        settings_for(tmp_path), provider=MatrixProvider(), now=lambda: NOW
    )
    assert pipeline.run().is_success
    second = pipeline.run()
    payload = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert len(payload["datasets"]) == 12
    for item in payload["datasets"]:
        frame = pd.read_parquet(item["file_path"])
        assert len(frame) == 3
        assert frame["timestamp"].is_unique


def test_one_failure_is_reported_after_other_pairs_continue(tmp_path: Path) -> None:
    failed_pair = ("SOLUSDT", "1d")
    provider = MatrixProvider(fail_pair=failed_pair)
    result = MultiAssetMarketDataPipeline(
        settings_for(tmp_path), provider=provider, now=lambda: NOW
    ).run()
    assert not result.is_success
    assert result.successful_datasets == 11
    assert len(provider.calls) == 12
    assert (result.failures[0].symbol, result.failures[0].interval) == failed_pair
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["last_run"]["status"] == "FAIL"
    assert payload["last_run"]["failed_datasets"] == 1
