"""Consistent multiclass validation metrics and winner selection."""

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.models.base import LABEL_ORDER


def evaluate_predictions(model_name: str, actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Calculate aggregate, per-class, and confusion-matrix metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=LABEL_ORDER, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    class_metrics = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABEL_ORDER)
    }
    matrix = confusion_matrix(actual, predicted, labels=LABEL_ORDER)
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "class_metrics": class_metrics,
        "confusion_matrix": {"labels": list(LABEL_ORDER), "matrix": matrix.tolist()},
    }


def select_best_model(results: list[dict[str, Any]], primary_metric: str = "macro_f1") -> dict[str, Any]:
    """Select the highest validation score without assumptions about model type."""
    if not results:
        raise ValueError("Cannot select a model from empty evaluation results")
    if any(primary_metric not in result for result in results):
        raise ValueError(f"Metric is missing from results: {primary_metric}")
    return max(results, key=lambda result: float(result[primary_metric]))
