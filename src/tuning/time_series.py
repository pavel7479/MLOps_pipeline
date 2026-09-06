"""Leakage-safe expanding-window cross-validation helpers."""

from typing import Any

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data.models import timeframe_to_timedelta


def derive_gap_rows(horizon_hours: int, timeframe: str) -> int:
    """Return the target horizon in rows; reject ambiguous timeframes."""
    horizon = pd.Timedelta(horizon_hours, unit="h")
    step = timeframe_to_timedelta(timeframe)
    periods = horizon / step
    if periods < 1 or not float(periods).is_integer():
        raise ValueError("Target horizon must be an exact multiple of the market timeframe")
    return int(periods)


def build_time_series_split(n_splits: int, gap: int) -> TimeSeriesSplit:
    """Build sklearn's chronological expanding-window splitter."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if gap < 0:
        raise ValueError("gap cannot be negative")
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def describe_folds(
    splitter: TimeSeriesSplit, timestamps: pd.Series
) -> list[dict[str, Any]]:
    """Record row and timestamp boundaries for auditing every fold."""
    folds: list[dict[str, Any]] = []
    for number, (train_indices, validation_indices) in enumerate(splitter.split(timestamps), start=1):
        folds.append({
            "fold": number,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "train_start_index": int(train_indices[0]),
            "train_end_index": int(train_indices[-1]),
            "validation_start_index": int(validation_indices[0]),
            "validation_end_index": int(validation_indices[-1]),
            "observed_gap_rows": int(validation_indices[0] - train_indices[-1] - 1),
            "train_start_timestamp": pd.Timestamp(timestamps.iloc[train_indices[0]]).isoformat(),
            "train_end_timestamp": pd.Timestamp(timestamps.iloc[train_indices[-1]]).isoformat(),
            "validation_start_timestamp": pd.Timestamp(timestamps.iloc[validation_indices[0]]).isoformat(),
            "validation_end_timestamp": pd.Timestamp(timestamps.iloc[validation_indices[-1]]).isoformat(),
        })
    return folds
