"""Data transfer objects used by the ingestion pipeline."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MARKET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


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

