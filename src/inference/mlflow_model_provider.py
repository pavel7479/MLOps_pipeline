"""MLflow implementation of the generic model provider."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import mlflow

from src.config import InferenceSettings, MLflowSettings
from src.mlops import MLflowTracker

from .model_provider import LoadedModel


class MLflowModelLoadError(RuntimeError):
    """Raised when the configured Registry alias cannot become an API model."""


class MLflowModelProvider:
    """Resolve alias metadata and load a signed MLflow pyfunc model once."""

    def __init__(
        self, mlflow_settings: MLflowSettings, inference_settings: InferenceSettings
    ) -> None:
        self.mlflow_settings = mlflow_settings
        self.inference_settings = inference_settings

    def _feature_names(self, model, client, run_id: str) -> tuple[str, ...]:
        schema = model.metadata.get_input_schema()
        if schema is not None:
            names = tuple(schema.input_names())
            if names:
                return names
        try:
            manifest_path = client.download_artifacts(
                run_id, "dataset/feature_manifest.json"
            )
            with open(manifest_path, encoding="utf-8") as source:
                names = tuple(json.load(source)["features"])
        except Exception as exc:
            raise MLflowModelLoadError(
                "Champion model has no usable feature signature or manifest"
            ) from exc
        if not names:
            raise MLflowModelLoadError("Champion model feature list is empty")
        return names

    def load_model(self) -> LoadedModel:
        tracker = MLflowTracker(self.mlflow_settings)
        tracker.check_health()
        mlflow.set_tracking_uri(self.mlflow_settings.tracking_uri)
        client = mlflow.MlflowClient()
        name = self.inference_settings.registered_model_name
        alias = self.inference_settings.model_alias
        uri = f"models:/{name}@{alias}"
        try:
            version = client.get_model_version_by_alias(name, alias)
            model = mlflow.pyfunc.load_model(uri)
            feature_names = self._feature_names(model, client, version.run_id)
        except Exception as exc:
            raise MLflowModelLoadError(
                f"Unable to load MLflow model alias: {name}@{alias}"
            ) from exc
        return LoadedModel(
            model=model,
            registered_model_name=name,
            alias=alias,
            version=str(version.version),
            run_id=str(version.run_id),
            feature_names=feature_names,
            loaded_at=datetime.now(timezone.utc),
            backtest_status=version.tags.get("backtest_status"),
        )
