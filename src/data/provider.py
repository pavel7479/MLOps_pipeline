"""Exchange-independent market data provider contract."""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class MarketDataProvider(ABC):
    """Source of normalized historical OHLCV candles."""

    @abstractmethod
    def fetch_ohlcv(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None
    ) -> pd.DataFrame:
        """Return candles in the canonical six-column representation."""

