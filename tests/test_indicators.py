import numpy as np
import pandas as pd

from src.features.indicators import (
    average_true_range,
    moving_average_convergence_divergence,
    relative_strength_index,
    true_range,
)


def test_rsi_warmup_range_and_trends() -> None:
    rising = pd.Series(np.arange(1.0, 31.0))
    falling = rising.iloc[::-1].reset_index(drop=True)
    rising_rsi = relative_strength_index(rising, 14)
    falling_rsi = relative_strength_index(falling, 14)
    assert rising_rsi.iloc[:14].isna().all()
    assert rising_rsi.dropna().between(0, 100).all()
    assert rising_rsi.iloc[-1] == 100
    assert falling_rsi.iloc[-1] == 0


def test_macd_matches_explicit_pandas_formula() -> None:
    close = pd.Series(np.linspace(100, 140, 50))
    actual = moving_average_convergence_divergence(close, 3, 6, 2)
    expected_line = close.ewm(span=3, adjust=False, min_periods=3).mean() - close.ewm(
        span=6, adjust=False, min_periods=6
    ).mean()
    expected_signal = expected_line.ewm(span=2, adjust=False, min_periods=2).mean()
    pd.testing.assert_series_equal(actual["macd"], expected_line, check_names=False)
    pd.testing.assert_series_equal(actual["macd_signal"], expected_signal, check_names=False)
    pd.testing.assert_series_equal(actual["macd_histogram"], expected_line - expected_signal, check_names=False)


def test_true_range_and_atr() -> None:
    frame = pd.DataFrame(
        {"high": [12.0, 15.0, 14.0], "low": [9.0, 11.0, 10.0], "close": [10.0, 14.0, 11.0]}
    )
    expected_true_range = pd.Series([3.0, 5.0, 4.0])
    pd.testing.assert_series_equal(true_range(frame), expected_true_range)
    assert average_true_range(frame, 2).iloc[0] != average_true_range(frame, 2).iloc[0]
    assert average_true_range(frame, 2).iloc[1] == 4.0
