"""Deterministic test-only classifier used by disposable CI registries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseClassifier, LABEL_MAPPING


class _ConstantEstimator:
    """Minimal fitted-like estimator; constructing it performs no training."""

    def __init__(self, encoded_label: int) -> None:
        self.encoded_label = encoded_label
        self.classes_ = np.asarray(sorted(LABEL_MAPPING.values()), dtype=int)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.encoded_label, dtype=int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.zeros((len(features), len(self.classes_)), dtype=float)
        probabilities[:, self.encoded_label] = 1.0
        return probabilities


class CIConstantModel(BaseClassifier):
    """Small CI fixture, never a trained or production model candidate."""

    name = "ci_fixture_constant"

    def __init__(self, label: str = "HOLD") -> None:
        if label not in LABEL_MAPPING:
            raise ValueError(f"Unsupported CI fixture label: {label}")
        parameters: dict[str, Any] = {
            "fixture": True,
            "constant_label": label,
            "trained": False,
        }
        super().__init__(_ConstantEstimator(LABEL_MAPPING[label]), parameters)

    def coefficient_report(
        self, feature_names: list[str]
    ) -> dict[str, list[dict[str, float | str]]]:
        return {}
