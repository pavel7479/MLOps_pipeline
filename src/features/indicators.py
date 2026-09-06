"""Technical indicators implemented with causal pandas operations."""

import pandas as pd


def relative_strength_index(close: pd.Series, period: int) -> pd.Series:
    """Calculate Wilder-style RSI using current and past closes only."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss
    result = 100 - (100 / (1 + relative_strength))
    result = result.mask((average_loss == 0) & (average_gain > 0), 100.0)
    result = result.mask((average_loss == 0) & (average_gain == 0), 50.0)
    return result


def moving_average_convergence_divergence(
    close: pd.Series, fast_period: int, slow_period: int, signal_period: int
) -> pd.DataFrame:
    """Calculate MACD line, signal, and histogram without future observations."""
    fast = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    slow = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    line = fast - slow
    signal = line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal, "macd_histogram": line - signal},
        index=close.index,
    )


def true_range(data: pd.DataFrame) -> pd.Series:
    """Calculate candle true range against the previous close."""
    previous_close = data["close"].shift(1)
    ranges = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def average_true_range(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate rolling ATR with a full lookback window."""
    return true_range(data).rolling(period, min_periods=period).mean()
