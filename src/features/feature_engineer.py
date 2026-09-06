"""Configuration-driven causal feature engineering."""

import numpy as np
import pandas as pd

from src.config.settings import FeaturesSettings

from .indicators import average_true_range, moving_average_convergence_divergence, relative_strength_index


class FeatureEngineer:
    """Create an explicit, stable set of numeric features from OHLCV data."""

    def __init__(self, settings: FeaturesSettings) -> None:
        self.settings = settings

    @property
    def feature_names(self) -> list[str]:
        """Return the manifest-safe feature list in deterministic order."""
        names = [f"return_{period}h" for period in self.settings.return_periods]
        for window in self.settings.rolling_mean_windows:
            names.extend((f"close_sma_{window}", f"close_to_sma_{window}"))
        names.extend(f"volatility_{window}" for window in self.settings.rolling_std_windows)
        names.extend(f"volume_change_{period}h" for period in self.settings.volume_change_periods)
        for window in self.settings.volume_rolling_windows:
            names.extend((f"volume_mean_{window}", f"volume_to_mean_{window}"))
        names.extend(("high_low_range", "body_size", f"rsi_{self.settings.rsi_period}"))
        names.extend(("macd", "macd_signal", "macd_histogram"))
        names.extend((f"atr_{self.settings.atr_period}", "atr_relative"))
        return names

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append features that depend only on values at or before each row."""
        result = data.copy()
        close = result["close"]
        for period in self.settings.return_periods:
            result[f"return_{period}h"] = close.pct_change(periods=period, fill_method=None)
        for window in self.settings.rolling_mean_windows:
            mean = close.rolling(window, min_periods=window).mean()
            result[f"close_sma_{window}"] = mean
            result[f"close_to_sma_{window}"] = close / mean - 1
        one_period_return = close.pct_change(fill_method=None)
        for window in self.settings.rolling_std_windows:
            result[f"volatility_{window}"] = one_period_return.rolling(window, min_periods=window).std()
        for period in self.settings.volume_change_periods:
            result[f"volume_change_{period}h"] = result["volume"].pct_change(periods=period, fill_method=None)
        for window in self.settings.volume_rolling_windows:
            mean = result["volume"].rolling(window, min_periods=window).mean()
            result[f"volume_mean_{window}"] = mean
            result[f"volume_to_mean_{window}"] = result["volume"] / mean
        result["high_low_range"] = (result["high"] - result["low"]) / close
        result["body_size"] = (close - result["open"]) / result["open"]
        result[f"rsi_{self.settings.rsi_period}"] = relative_strength_index(close, self.settings.rsi_period)
        macd = moving_average_convergence_divergence(
            close,
            self.settings.macd_fast_period,
            self.settings.macd_slow_period,
            self.settings.macd_signal_period,
        )
        result[["macd", "macd_signal", "macd_histogram"]] = macd
        atr_name = f"atr_{self.settings.atr_period}"
        result[atr_name] = average_true_range(result, self.settings.atr_period)
        result["atr_relative"] = result[atr_name] / close
        result[self.feature_names] = result[self.feature_names].replace([np.inf, -np.inf], np.nan)
        return result
