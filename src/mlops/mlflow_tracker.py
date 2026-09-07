"""Small, explicit facade around MLflow Experiment Tracking."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from src.config.settings import MLflowSettings
from src.models import BaseClassifier

from .models import CryptoClassifierPyfunc

LOGGER = logging.getLogger(__name__)


class MLflowTrackingError(RuntimeError):
    """Raised when enabled tracking cannot be reached or used."""


class MLflowTracker:
    """Centralized optional MLflow tracking operations."""

    def __init__(self, settings: MLflowSettings) -> None:
        # MLflow 3 prints emoji run/model URLs that fail on Windows cp1251 consoles.
        os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
        os.environ.setdefault("MLFLOW_PRINT_MODEL_URLS_ON_CREATION", "false")
        self.settings = settings
        self._experiment_id: str | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def experiment_id(self) -> str | None:
        return self._experiment_id

    def check_health(self, timeout_seconds: float = 2.0) -> None:
        """Fail fast for an unavailable HTTP server; allow direct local stores."""
        if not self.enabled:
            return
        if self.settings.tracking_uri.lower().startswith(("http://", "https://")):
            url = f"{self.settings.tracking_uri.rstrip('/')}/health"
            try:
                response = httpx.get(url, timeout=timeout_seconds)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise MLflowTrackingError(
                    "MLflow tracking server is unavailable: "
                    f"{self.settings.tracking_uri}"
                ) from exc
        try:
            self._configure()
        except Exception as exc:
            raise MLflowTrackingError(
                f"MLflow tracking server is unavailable: {self.settings.tracking_uri}"
            ) from exc

    def _configure(self) -> None:
        if not self.enabled:
            return
        mlflow.set_tracking_uri(self.settings.tracking_uri)
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self.settings.experiment_name)
        if experiment is None:
            self._experiment_id = client.create_experiment(
                self.settings.experiment_name,
                artifact_location=self.settings.artifact_location,
            )
        else:
            self._experiment_id = experiment.experiment_id

    def start_run(self, run_name: str, tags: dict[str, Any] | None = None) -> str | None:
        if not self.enabled:
            return None
        if self._experiment_id is None:
            self._configure()
        run = mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
            tags={key: str(value) for key, value in (tags or {}).items()},
        )
        return run.info.run_id

    def log_params(self, parameters: dict[str, Any]) -> None:
        if not self.enabled:
            return
        clean = {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
            for key, value in parameters.items()
        }
        mlflow.log_params(clean)

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        if not self.enabled:
            return
        clean = {
            key: float(value)
            for key, value in metrics.items()
            if np.isfinite(float(value))
        }
        mlflow.log_metrics(clean)

    def log_tags(self, tags: dict[str, Any]) -> None:
        if self.enabled:
            mlflow.set_tags({key: str(value) for key, value in tags.items()})

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"MLflow artifact not found: {source}")
        mlflow.log_artifact(str(source), artifact_path=artifact_path)

    def log_dict(
        self, payload: dict[str, Any] | list[Any], artifact_file: str
    ) -> None:
        if self.enabled:
            mlflow.log_dict(payload, artifact_file)

    def log_model(
        self,
        classifier: BaseClassifier,
        feature_names: list[str],
        input_example: pd.DataFrame,
        *,
        artifact_name: str = "model",
        code_paths: list[str] | None = None,
    ) -> tuple[str, Any]:
        """Log a signed pyfunc model and verify predictions after MLflow load."""
        if not self.enabled:
            return "", np.asarray([])
        example = input_example[feature_names].copy()
        expected = classifier.predict(example)
        python_model = CryptoClassifierPyfunc(classifier, feature_names)
        signature = infer_signature(example, expected)
        info = mlflow.pyfunc.log_model(
            name=artifact_name,
            python_model=python_model,
            signature=signature,
            input_example=example,
            code_paths=code_paths,
        )
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        actual = np.asarray(loaded.predict(example))
        if not np.array_equal(expected, actual):
            raise MLflowTrackingError("Predictions changed after MLflow model load")
        return info.model_uri, actual

    def end_run(self, status: str = "FINISHED") -> None:
        if self.enabled and mlflow.active_run() is not None:
            mlflow.end_run(status=status)
