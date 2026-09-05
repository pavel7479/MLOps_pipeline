from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "open": [10.0, 11.0, 12.0], "high": [12.0, 13.0, 14.0],
        "low": [9.0, 10.0, 11.0], "close": [11.0, 12.0, 13.0], "volume": [1.0, 2.0, 3.0],
    })


def kline(hour: int) -> list[object]:
    stamp = int(datetime(2024, 1, 1, hour, tzinfo=timezone.utc).timestamp() * 1000)
    return [stamp, "10", "12", "9", "11", "5", stamp + 1, "0", 1, "0", "0", "0"]
