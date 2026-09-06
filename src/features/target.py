"""Future-return classification target construction."""

import numpy as np
import pandas as pd

from src.config.settings import TargetSettings

TARGET_CLASSES = ("BUY", "HOLD", "SELL")


class TargetBuilder:
    """Build a three-class target from a strictly future close."""

    def __init__(self, settings: TargetSettings) -> None:
        self.settings = settings

    def transform(self, data: pd.DataFrame, horizon_periods: int) -> pd.DataFrame:
        """Append future_return and target; unavailable final targets stay null."""
        result = data.copy()
        future_close = result["close"].shift(-horizon_periods)
        result["future_return"] = future_close / result["close"] - 1
        target = pd.Series(pd.NA, index=result.index, dtype="string")
        available = result["future_return"].notna()
        values = np.select(
            [
                result.loc[available, "future_return"] >= self.settings.buy_threshold,
                result.loc[available, "future_return"] <= self.settings.sell_threshold,
            ],
            ["BUY", "SELL"],
            default="HOLD",
        )
        target.loc[available] = values
        result["target"] = target
        return result
