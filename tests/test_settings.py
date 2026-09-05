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
