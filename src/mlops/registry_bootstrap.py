"""Idempotently restore the two inference aliases from saved model artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import joblib
import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
import pandas as pd

from src.config import AppSettings
from src.models import BaseClassifier

from .mlflow_tracker import MLflowTracker
from .model_registry import ModelRegistry


@dataclass(frozen=True)
class BootstrapSource:
    alias: str
    run_name: str
    model_path: Path
    source_stage: str
    bootstrap_key: str


@dataclass(frozen=True)
class BootstrapResult:
    aliases: dict[str, str]
    created_aliases: tuple[str, ...]


class RegistryGateway(Protocol):
    def get_alias(self, alias: str) -> ModelVersion | None:
        """Return an alias target, or None when it does not exist."""

    def set_alias(self, alias: str, version: str) -> None:
        """Point an alias at an existing version."""


class MLflowRegistryGateway:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def get_alias(self, alias: str) -> ModelVersion | None:
        try:
            return self.registry.get_alias(alias)
        except MlflowException as exc:
            if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
                return None
            raise

    def set_alias(self, alias: str, version: str) -> None:
        self.registry.set_alias(alias, version)


def ensure_registry_aliases(
    gateway: RegistryGateway,
    sources: tuple[BootstrapSource, ...],
    manifest_path: Path,
    register_source: Callable[[BootstrapSource, Path], ModelVersion],
) -> BootstrapResult:
    """Create only missing aliases; validate every needed file before side effects."""
    aliases: dict[str, str] = {}
    missing: list[BootstrapSource] = []
    for source in sources:
        current = gateway.get_alias(source.alias)
        if current is None:
            missing.append(source)
        else:
            aliases[source.alias] = str(current.version)
    if not missing:
        return BootstrapResult(aliases=aliases, created_aliases=())

    required = (manifest_path, *(source.model_path for source in missing))
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError(
            "Cannot initialize MLflow Registry; required bootstrap artifact(s) "
            f"missing: {', '.join(absent)}"
        )

    created: list[str] = []
    for source in missing:
        version = register_source(source, manifest_path)
        gateway.set_alias(source.alias, str(version.version))
        aliases[source.alias] = str(version.version)
        created.append(source.alias)
    return BootstrapResult(aliases=aliases, created_aliases=tuple(created))


def initialize_registry(
    settings: AppSettings,
    *,
    baseline_model_path: Path,
    tuned_model_path: Path,
    manifest_path: Path,
) -> BootstrapResult:
    """Check aliases and bootstrap missing versions without training or datasets."""
    tracker = MLflowTracker(settings.mlflow)
    tracker.check_health()
    if tracker.experiment_id is None:
        raise RuntimeError("MLflow must be enabled for Registry initialization")
    registry = ModelRegistry(settings.mlflow)
    gateway = MLflowRegistryGateway(registry)
    sources = (
        BootstrapSource(
            alias="champion",
            run_name="docker_bootstrap_baseline_lightgbm",
            model_path=baseline_model_path,
            source_stage="block_3",
            bootstrap_key="baseline_lightgbm_block_3",
        ),
        BootstrapSource(
            alias="challenger",
            run_name="docker_bootstrap_tuned_lightgbm",
            model_path=tuned_model_path,
            source_stage="block_4",
            bootstrap_key="tuned_lightgbm_block_4",
        ),
    )

    def register_source(
        source: BootstrapSource, feature_manifest_path: Path
    ) -> ModelVersion:
        manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        feature_names = manifest.get("features")
        if not isinstance(feature_names, list) or not feature_names or not all(
            isinstance(name, str) for name in feature_names
        ):
            raise ValueError(
                "Cannot initialize MLflow Registry: feature manifest is invalid"
            )
        classifier = joblib.load(source.model_path)
        if not isinstance(classifier, BaseClassifier):
            raise TypeError(
                "Cannot initialize MLflow Registry: "
                f"unexpected model type {type(classifier).__name__}"
            )
        example = pd.DataFrame(
            [{feature: 0.0 for feature in feature_names}],
            columns=feature_names,
        )
        run_id = tracker.start_run(
            source.run_name,
            {
                "project": "crypto_ml_platform",
                "pipeline_stage": "block_8_registry_bootstrap",
                "source_stage": source.source_stage,
                "model_family": "lightgbm",
                "bootstrap_key": source.bootstrap_key,
                "test_dataset_used": "false",
            },
        )
        if run_id is None:
            raise RuntimeError("MLflow bootstrap could not create a run")
        try:
            tracker.log_params(
                {
                    "features_count": len(feature_names),
                    "bootstrap_without_training": True,
                    **classifier.parameters,
                }
            )
            tracker.log_artifact(feature_manifest_path, "dataset")
            tracker.log_artifact(source.model_path, "native_model")
            tracker.log_model(
                classifier,
                feature_names,
                example,
                code_paths=[str(Path(__file__).resolve().parents[1])],
            )
        except Exception:
            tracker.end_run("FAILED")
            raise
        tracker.end_run()
        return registry.register_run_model(
            run_id,
            description=(
                f"Restored from saved {source.source_stage} artifact by the "
                "idempotent Block 8 container bootstrap; no training performed."
            ),
            tags={
                "source_stage": source.source_stage,
                "model_type": "lightgbm",
                "validation_status": "passed",
                "backtest_status": "failed_profitability_check",
                "bootstrap_key": source.bootstrap_key,
            },
        )

    return ensure_registry_aliases(
        gateway, sources, manifest_path, register_source
    )
