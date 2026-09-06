"""Validation rules for complete ML datasets and temporal splits."""

import numpy as np
import pandas as pd

from src.data.models import MARKET_COLUMNS
from src.features.target import TARGET_CLASSES

from .models import DatasetSplits


class MLDatasetValidator:
    """Fail fast on leakage-prone schema, invalid features, or split overlap."""

    FORBIDDEN_FEATURES = {"future_return", "future_close", "target"}

    def validate_dataset(self, data: pd.DataFrame, feature_names: list[str]) -> None:
        """Validate a final, training-ready ML dataset."""
        if data.empty:
            raise ValueError("ML dataset is empty")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Feature manifest contains duplicate names")
        forbidden = self.FORBIDDEN_FEATURES.intersection(feature_names)
        forbidden.update(name for name in feature_names if name.startswith("future_"))
        if forbidden:
            raise ValueError(f"Future or target fields found in features: {sorted(forbidden)}")
        required = [*MARKET_COLUMNS, *feature_names, "future_return", "target"]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"ML dataset is missing columns: {missing}")
        if data["timestamp"].isna().any() or not data["timestamp"].is_unique:
            raise ValueError("ML timestamps must be non-null and unique")
        if not data["timestamp"].is_monotonic_increasing:
            raise ValueError("ML timestamps must be sorted")
        numeric = data[[*feature_names, "future_return"]]
        if not all(pd.api.types.is_numeric_dtype(numeric[column]) for column in numeric):
            raise ValueError("All features and future_return must be numeric")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("Features and future_return must be finite and non-null")
        if data[MARKET_COLUMNS].isna().any().any():
            raise ValueError("Required OHLCV values cannot be null")
        invalid_targets = set(data["target"].dropna().unique()) - set(TARGET_CLASSES)
        if data["target"].isna().any() or invalid_targets:
            raise ValueError(f"Invalid target values: {sorted(invalid_targets)}")

    def validate_splits(self, splits: DatasetSplits) -> None:
        """Validate non-empty, ordered, non-overlapping temporal partitions."""
        parts = (splits.train, splits.validation, splits.test)
        if any(part.empty for part in parts):
            raise ValueError("Train, validation, and test splits must be non-empty")
        if not all(part["timestamp"].is_monotonic_increasing for part in parts):
            raise ValueError("Every split must remain chronologically sorted")
        if not splits.train["timestamp"].max() < splits.validation["timestamp"].min():
            raise ValueError("Train and validation time ranges overlap")
        if not splits.validation["timestamp"].max() < splits.test["timestamp"].min():
            raise ValueError("Validation and test time ranges overlap")
        timestamp_sets = [set(part["timestamp"]) for part in parts]
        if timestamp_sets[0] & timestamp_sets[1] or timestamp_sets[1] & timestamp_sets[2] or timestamp_sets[0] & timestamp_sets[2]:
            raise ValueError("Split timestamps overlap")
