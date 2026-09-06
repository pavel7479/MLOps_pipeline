from pathlib import Path

import pytest

from src.config.settings import ConfigurationError, load_settings


def test_loads_yaml_and_resolves_paths(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("""market_data: {provider: binance, symbol: ethusdt, timeframe: 1h, start_date: '2024-01-01', end_date: null}
storage: {raw_dir: data/raw, processed_dir: data/processed, format: parquet}
http: {timeout_seconds: 1, max_retries: 2, retry_delay_seconds: 0}
logging: {level: DEBUG}
""", encoding="utf-8")
    settings = load_settings(config)
    assert settings.market_data.symbol == "ETHUSDT"
    assert settings.market_data.timeframe == "1h"
    assert settings.storage.raw_dir == (tmp_path / "data/raw").resolve()


def test_missing_required_setting_is_clear(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("market_data: {}\nstorage: {}\nhttp: {}\nlogging: {}", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="start_date"):
        load_settings(path)


def test_block_two_settings_are_loaded() -> None:
    settings = load_settings(Path(__file__).resolve().parents[1] / "config.yaml")
    assert settings.features.return_periods == (1, 3, 6, 12, 24)
    assert settings.target.horizon_hours == 3
    assert settings.ml_dataset.output_dir.name == "ml"


def test_split_ratios_must_sum_to_one(tmp_path: Path) -> None:
    source = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid-ratios.yaml"
    path.write_text(source.replace("train_ratio: 0.70", "train_ratio: 0.80"), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="sum to 1.0"):
        load_settings(path)
