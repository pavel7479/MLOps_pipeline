"""Data transfer objects used by the ingestion pipeline."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd



MARKET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convert a fixed-duration Binance timeframe to an explicit timedelta."""
    units = {"s": "s", "m": "min", "h": "h", "d": "D", "w": "W"}
    try:
        count = int(timeframe[:-1])
        unit = units[timeframe[-1]]
    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"Unsupported fixed-duration timeframe: {timeframe}") from exc
    return pd.Timedelta(count, unit=unit)



@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    invalid_rows: int
    duplicate_timestamps: int
    missing_intervals: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataQualityReport:
    downloaded_rows: int
    duplicate_timestamps: int
    invalid_rows: int
    missing_intervals: int
    final_rows: int
    start_timestamp: datetime | None
    end_timestamp: datetime | None
    raw_path: Path
    processed_path: Path

