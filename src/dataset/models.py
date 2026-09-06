"""Data structures returned by ML dataset preparation."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    purged_rows: int


@dataclass(frozen=True)
class MLDatasetReport:
    input_rows: int
    final_rows: int
    feature_count: int
    rows_removed_for_features: int
    rows_removed_for_target: int
    target_distribution: dict[str, dict[str, float | int]]
    train_rows: int
    validation_rows: int
    test_rows: int
    train_range: tuple[datetime, datetime]
    validation_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]
    purged_rows: int
    dataset_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    manifest_path: Path
    metadata_path: Path
    statistics_path: Path
