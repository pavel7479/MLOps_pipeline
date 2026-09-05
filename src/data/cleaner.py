"""Deterministic and idempotent cleaning of OHLCV frames."""

import pandas as pd

from .models import MARKET_COLUMNS


class MarketDataCleaner:
    """Normalize types and remove repair-unsafe rows without fabrication."""

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in MARKET_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"Cannot clean data missing columns: {', '.join(missing)}")
        frame = data[MARKET_COLUMNS].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        numeric_columns = MARKET_COLUMNS[1:]
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=MARKET_COLUMNS)
        valid = (frame[["open", "high", "low", "close"]] > 0).all(axis=1) & (frame["volume"] >= 0)
        valid &= frame["high"] >= frame[["open", "low", "close"]].max(axis=1)
        valid &= frame["low"] <= frame[["open", "close"]].min(axis=1)
        return frame.loc[valid].drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
