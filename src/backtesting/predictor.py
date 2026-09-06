"""Prediction adapter for already-fitted project model artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .engine import ALLOWED_SIGNALS


@dataclass(frozen=True)
class PredictionOutput:
    predictions: np.ndarray
    model_name: str
    model_class: str
    parameters: dict[str, Any]


class ArtifactPredictor:
    """Load an artifact, validate feature compatibility, and predict."""

    def predict(
        self, model_path: Path, validation: pd.DataFrame, feature_names: list[str]
    ) -> PredictionOutput:
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved model artifact not found: {model_path}")
        missing = [feature for feature in feature_names if feature not in validation]
        if missing:
            raise ValueError(f"Validation is missing model features: {missing}")
        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            raise TypeError(f"Artifact does not provide predict(): {model_path}")
        estimator = getattr(model, "estimator", model)
        expected_count = getattr(estimator, "n_features_in_", None)
        if expected_count is not None and int(expected_count) != len(feature_names):
            raise ValueError(
                f"Model expects {expected_count} features, manifest contains {len(feature_names)}"
            )
        fitted_names = getattr(estimator, "feature_name_", None)
        if fitted_names is None:
            fitted_names = getattr(estimator, "feature_names_in_", None)
        if fitted_names is not None and list(fitted_names) != feature_names:
            raise ValueError("Model feature order is incompatible with feature_manifest.json")
        predictions = np.asarray(model.predict(validation[feature_names])).reshape(-1)
        invalid = set(predictions) - ALLOWED_SIGNALS
        if len(predictions) != len(validation) or invalid:
            raise ValueError(f"Model returned invalid predictions: {sorted(invalid)}")
        return PredictionOutput(
            predictions=predictions.astype(str),
            model_name=str(getattr(model, "name", type(model).__name__)),
            model_class=type(model).__name__,
            parameters=dict(getattr(model, "parameters", {})),
        )
