"""Explicit lifecycle operations for the single classifier Model Registry entry."""

from __future__ import annotations

import os
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion

from src.config.settings import MLflowSettings


REGISTERED_MODEL_DESCRIPTION = (
    "BTCUSDT 1h multiclass classifier. Predicts BUY/HOLD/SELL. "
    "Educational ML/MLOps project. No real-money trading."
)


class ModelRegistry:
    """Manage model versions and aliases without automatic promotion."""

    def __init__(self, settings: MLflowSettings) -> None:
        os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
        os.environ.setdefault("MLFLOW_PRINT_MODEL_URLS_ON_CREATION", "false")
        self.settings = settings
        mlflow.set_tracking_uri(settings.tracking_uri)
        self.client = mlflow.MlflowClient()

    def ensure_registered_model(self) -> None:
        try:
            self.client.get_registered_model(self.settings.registered_model_name)
        except mlflow.exceptions.MlflowException:
            self.client.create_registered_model(
                self.settings.registered_model_name,
                description=REGISTERED_MODEL_DESCRIPTION,
            )
        else:
            self.client.update_registered_model(
                self.settings.registered_model_name,
                description=REGISTERED_MODEL_DESCRIPTION,
            )

    def register_run_model(
        self,
        run_id: str,
        *,
        description: str,
        tags: dict[str, Any],
        artifact_name: str = "model",
    ) -> ModelVersion:
        """Register once per run and return the actual Registry version."""
        self.ensure_registered_model()
        for existing in self.client.search_model_versions(
            f"name='{self.settings.registered_model_name}'"
        ):
            if existing.run_id == run_id:
                version = existing
                break
        else:
            version = mlflow.register_model(
                f"runs:/{run_id}/{artifact_name}",
                self.settings.registered_model_name,
            )
        self.client.update_model_version(
            self.settings.registered_model_name,
            version.version,
            description=description,
        )
        for key, value in tags.items():
            self.client.set_model_version_tag(
                self.settings.registered_model_name,
                version.version,
                key,
                str(value),
            )
        return version

    def set_champion(self, model_version: str | int) -> None:
        self.set_alias("champion", model_version)

    def set_challenger(self, model_version: str | int) -> None:
        self.set_alias("challenger", model_version)

    def set_alias(self, alias: str, model_version: str | int) -> None:
        self.client.set_registered_model_alias(
            self.settings.registered_model_name, alias, str(model_version)
        )

    def get_alias(self, alias: str) -> ModelVersion:
        return self.client.get_model_version_by_alias(
            self.settings.registered_model_name, alias
        )

    def load_model_by_alias(self, alias: str):
        uri = f"models:/{self.settings.registered_model_name}@{alias}"
        return mlflow.pyfunc.load_model(uri)

    def load_champion_model(self):
        return self.load_model_by_alias("champion")


def load_champion_model(settings: MLflowSettings):
    """Load the active candidate without coupling inference to a version number."""
    return ModelRegistry(settings).load_champion_model()
