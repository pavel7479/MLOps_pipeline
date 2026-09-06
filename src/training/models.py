"""Structured result returned by the model training pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelTrainingReport:
    train_rows: int
    validation_rows: int
    features_count: int
    results: list[dict[str, Any]]
    best_model: str
    best_score: float
    dummy_score: float
    artifact_paths: dict[str, Path]
