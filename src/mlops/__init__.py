"""MLflow Experiment Tracking and Model Registry integration."""

from .mlflow_tracker import MLflowTracker, MLflowTrackingError
from .model_registry import ModelRegistry, load_champion_model
from .models import CryptoClassifierPyfunc, dataframe_period, git_metadata, sha256_file
from .registry_bootstrap import (
    BootstrapResult,
    BootstrapSource,
    ensure_registry_aliases,
    initialize_registry,
)

__all__ = [
    "CryptoClassifierPyfunc",
    "BootstrapResult",
    "BootstrapSource",
    "MLflowTracker",
    "MLflowTrackingError",
    "ModelRegistry",
    "dataframe_period",
    "ensure_registry_aliases",
    "git_metadata",
    "load_champion_model",
    "initialize_registry",
    "sha256_file",
]
