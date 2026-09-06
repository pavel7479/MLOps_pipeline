"""Leakage and data-quality checks before model fitting."""

import numpy as np
import pandas as pd

from src.features.target import TARGET_CLASSES


class TrainingDataValidator:
    """Validate shared Train/Validation inputs and an explicit feature manifest."""

    FORBIDDEN_FEATURES = {"timestamp", "open", "high", "low", "close", "volume", "future_return", "target"}

    def validate(self, train: pd.DataFrame, validation: pd.DataFrame, feature_names: list[str]) -> None:
        """Raise a clear error before any estimator sees invalid data."""
        if train.empty or validation.empty:
            raise ValueError("Train and validation datasets must be non-empty")
        if not feature_names or len(feature_names) != len(set(feature_names)):
            raise ValueError("Feature manifest must be non-empty and unique")
        forbidden = self.FORBIDDEN_FEATURES.intersection(feature_names)
        forbidden.update(name for name in feature_names if name.startswith("future_"))
        if forbidden:
            raise ValueError(f"Forbidden fields in feature manifest: {sorted(forbidden)}")
        for name, frame in (("train", train), ("validation", validation)):
            missing = [column for column in [*feature_names, "target", "timestamp"] if column not in frame]
            if missing:
                raise ValueError(f"{name} is missing required columns: {missing}")
            features = frame[feature_names]
            if not all(pd.api.types.is_numeric_dtype(features[column]) for column in feature_names):
                raise ValueError(f"{name} features must all be numeric")
            if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
                raise ValueError(f"{name} features must be finite and non-null")
            labels = set(frame["target"].dropna().unique())
            if frame["target"].isna().any() or not labels.issubset(TARGET_CLASSES):
                raise ValueError(f"{name} contains invalid target labels")
            if frame["timestamp"].isna().any() or not frame["timestamp"].is_unique:
                raise ValueError(f"{name} timestamps must be non-null and unique")
        train_timestamps = set(train["timestamp"])
        validation_timestamps = set(validation["timestamp"])
        if train_timestamps & validation_timestamps:
            raise ValueError("Train and validation timestamps overlap")
        if train["timestamp"].max() >= validation["timestamp"].min():
            raise ValueError("Train must end before validation starts")
