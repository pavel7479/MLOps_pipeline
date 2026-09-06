"""Structured results for hyperparameter tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ModelSearchResult:
    model: str
    best_params: dict[str, Any]
    mean_cv_macro_f1: float
    std_cv_macro_f1: float
    fold_scores: list[float]
    duration_seconds: float
    candidates: pd.DataFrame


@dataclass(frozen=True)
class TuningReport:
    train_rows: int
    validation_rows: int
    features_count: int
    n_splits: int
    gap_rows: int
    cv_results: list[dict[str, Any]]
    validation_results: list[dict[str, Any]]
    best_model: str
    best_score: float
    artifact_paths: dict[str, Path]
