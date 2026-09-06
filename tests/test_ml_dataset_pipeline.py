import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import (
    AppSettings, FeaturesSettings, HttpSettings, LoggingSettings, MarketDataSettings,
    MLDatasetSettings, SplitSettings, StorageSettings, TargetSettings,
)
from src.dataset.pipeline import MLDatasetPipeline


def test_ml_dataset_pipeline_creates_all_artifacts(tmp_path: Path) -> None:
    size = 240
    close = 100 + np.sin(np.arange(size) / 3) * 2 + np.arange(size) * 0.01
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=size, freq="h", tz="UTC"),
            "open": close - 0.1, "high": close + 1, "low": close - 1,
            "close": close, "volume": 20 + np.cos(np.arange(size) / 4),
        }
    )
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    source.to_parquet(processed_dir / "BTCUSDT_1h.parquet", index=False)
    features = FeaturesSettings(
        return_periods=(1, 3), rolling_mean_windows=(3, 6), rolling_std_windows=(3,),
        volume_change_periods=(1,), volume_rolling_windows=(3,), rsi_period=3,
        macd_fast_period=3, macd_slow_period=6, macd_signal_period=2, atr_period=3,
    )
    settings = AppSettings(
        MarketDataSettings("binance", "BTCUSDT", "1h", datetime(2024, 1, 1, tzinfo=timezone.utc), None),
        StorageSettings(tmp_path / "raw", processed_dir, "parquet"),
        HttpSettings(1, 0, 0), LoggingSettings("INFO"), features,
        TargetSettings(horizon_hours=3, buy_threshold=0.003, sell_threshold=-0.003),
        SplitSettings(), MLDatasetSettings(tmp_path / "ml"),
    )
    report = MLDatasetPipeline(settings).run()
    paths = [
        report.dataset_path, report.train_path, report.validation_path, report.test_path,
        report.manifest_path, report.metadata_path, report.statistics_path,
    ]
    assert all(path.is_file() for path in paths)
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["features"] == MLDatasetPipeline(settings).feature_engineer.feature_names
    assert "future_return" not in manifest["features"] and "target" not in manifest["features"]
    assert report.final_rows < report.input_rows
    assert report.train_rows + report.validation_rows + report.test_rows == report.final_rows - report.purged_rows
    assert sum(item["count"] for item in report.target_distribution.values()) == report.final_rows
