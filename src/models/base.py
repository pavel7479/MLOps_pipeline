"""Common classifier contract and stable target encoding."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd

LABEL_MAPPING = {"SELL": 0, "HOLD": 1, "BUY": 2}
INVERSE_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}
LABEL_ORDER = tuple(LABEL_MAPPING)


class BaseClassifier(ABC):
    """Uniform fit, inference, probability, and persistence interface."""

    name: str

    def __init__(self, estimator: Any, parameters: dict[str, Any]) -> None:
        self.estimator = estimator
        self.parameters = parameters

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "BaseClassifier":
        """Fit on numeric features using the project-wide label mapping."""
        unknown = set(target.unique()) - set(LABEL_MAPPING)
        if unknown:
            raise ValueError(f"Unsupported target labels: {sorted(unknown)}")
        self.estimator.fit(features, target.map(LABEL_MAPPING).astype(int))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return predictions as BUY/HOLD/SELL strings."""
        encoded = np.asarray(self.estimator.predict(features)).reshape(-1)
        return np.asarray([INVERSE_LABEL_MAPPING[int(value)] for value in encoded])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return columns ordered as P(SELL), P(HOLD), P(BUY)."""
        raw = np.asarray(self.estimator.predict_proba(features))
        aligned = np.zeros((len(features), len(LABEL_MAPPING)), dtype=float)
        for source_column, encoded_class in enumerate(self.estimator.classes_):
            aligned[:, int(encoded_class)] = raw[:, source_column]
        return aligned

    def save(self, path: str | Path) -> Path:
        """Persist the complete wrapper and fitted estimator with joblib."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump(self, temporary)
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a wrapper and verify its concrete type."""
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded

    def feature_importance(self, feature_names: list[str]) -> list[dict[str, float | str]]:
        """Return descending built-in feature importance when supported."""
        values = getattr(self.estimator, "feature_importances_", None)
        if values is None:
            return []
        rows = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_names, values, strict=True)
        ]
        return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)

    @abstractmethod
    def coefficient_report(self, feature_names: list[str]) -> dict[str, list[dict[str, float | str]]]:
        """Return coefficients for linear models and an empty mapping otherwise."""
