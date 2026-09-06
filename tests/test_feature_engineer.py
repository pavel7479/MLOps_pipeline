import numpy as np
import pandas as pd
import pytest

from src.config.settings import FeaturesSettings
from src.features.feature_engineer import FeatureEngineer


def market_frame(size: int = 60) -> pd.DataFrame:
    close = pd.Series(np.linspace(100.0, 130.0, size))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=size, freq="h", tz="UTC"),
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(10.0, 30.0, size),
        }
    )


def compact_settings() -> FeaturesSettings:
    return FeaturesSettings(
        return_periods=(1, 3), rolling_mean_windows=(3,), rolling_std_windows=(3,),
        volume_change_periods=(1,), volume_rolling_windows=(3,), rsi_period=3,
        macd_fast_period=3, macd_slow_period=5, macd_signal_period=2, atr_period=3,
    )


def test_returns_sma_and_volatility_match_pandas() -> None:
    source = market_frame()
    result = FeatureEngineer(compact_settings()).transform(source)
    assert result.loc[3, "return_1h"] == source.loc[3, "close"] / source.loc[2, "close"] - 1
    assert result.loc[3, "return_3h"] == source.loc[3, "close"] / source.loc[0, "close"] - 1
    assert result.loc[2, "close_sma_3"] == pytest.approx(source.loc[:2, "close"].mean())
    expected_volatility = source["close"].pct_change(fill_method=None).rolling(3).std()
    pd.testing.assert_series_equal(result["volatility_3"], expected_volatility, check_names=False)


def test_future_changes_do_not_change_past_features() -> None:
    engineer = FeatureEngineer(compact_settings())
    original = market_frame()
    cutoff = 30
    before = engineer.transform(original).loc[cutoff, engineer.feature_names]
    changed = original.copy()
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close", "volume"]] *= 10
    after = engineer.transform(changed).loc[cutoff, engineer.feature_names]
    pd.testing.assert_series_equal(before, after)


def test_feature_manifest_is_explicit_and_has_no_future_fields() -> None:
    names = FeatureEngineer(compact_settings()).feature_names
    assert "return_1h" in names and "rsi_3" in names and "atr_relative" in names
    assert not {"future_return", "future_close", "target"}.intersection(names)
