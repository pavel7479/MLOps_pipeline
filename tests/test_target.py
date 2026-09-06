import pandas as pd
import pytest

from src.config.settings import TargetSettings
from src.features.target import TargetBuilder


def test_future_return_shift_and_target_classes() -> None:
    source = pd.DataFrame({"close": [100.0, 101.0, 100.0, 100.0, 102.0]})
    builder = TargetBuilder(TargetSettings(horizon_hours=1, buy_threshold=0.005, sell_threshold=-0.005))
    result = builder.transform(source, horizon_periods=1)
    expected = source["close"].shift(-1) / source["close"] - 1
    pd.testing.assert_series_equal(result["future_return"], expected, check_names=False)
    assert result["target"].tolist() == ["BUY", "SELL", "HOLD", "BUY", pd.NA]


def test_horizon_uses_requested_future_row() -> None:
    source = pd.DataFrame({"close": [100.0, 110.0, 120.0, 130.0]})
    result = TargetBuilder(TargetSettings(horizon_hours=2)).transform(source, horizon_periods=2)
    assert result.loc[0, "future_return"] == pytest.approx(0.2)
    assert result["target"].iloc[-2:].isna().all()
