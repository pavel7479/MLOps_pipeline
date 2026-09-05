"""Market data quality validation and gap diagnostics."""

import pandas as pd

from .models import MARKET_COLUMNS, ValidationResult


class MarketDataValidator:
    """Validate schema, values, chronology, and candle continuity."""

    def validate(self, data: pd.DataFrame, timeframe: str) -> ValidationResult:
        missing = [column for column in MARKET_COLUMNS if column not in data.columns]
        if missing:
            return ValidationResult(False, len(data), 0, 0, (f"Missing columns: {', '.join(missing)}",))
        frame = data[MARKET_COLUMNS].copy()
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        numeric = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        duplicate_count = int(timestamps.duplicated(keep=False).sum())
        invalid = timestamps.isna() | numeric.isna().any(axis=1)
        invalid |= (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1) | (numeric["volume"] < 0)
        invalid |= (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1))
        invalid |= (numeric["low"] > numeric[["open", "close"]].min(axis=1))
        errors: list[str] = []
        if duplicate_count:
            errors.append(f"Duplicate timestamps: {duplicate_count}")
        if not timestamps.is_monotonic_increasing:
            errors.append("Timestamps are not sorted")
        invalid_count = int(invalid.sum())
        if invalid_count:
            errors.append(f"Invalid rows: {invalid_count}")
        valid_unique = timestamps.dropna().drop_duplicates().sort_values()
        expected = pd.to_timedelta(timeframe)
        gaps = valid_unique.diff().dropna()
        missing_intervals = int(sum(max(0, int(delta / expected) - 1) for delta in gaps))
        if missing_intervals:
            errors.append(f"Missing intervals: {missing_intervals}")
        return ValidationResult(not errors, invalid_count, duplicate_count, missing_intervals, tuple(errors))
