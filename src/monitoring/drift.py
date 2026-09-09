"""Population Stability Index calculation for recent inference features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import MonitoringReference


@dataclass(frozen=True)
class DriftResult:
    available: bool
    status: str
    scores: dict[str, float]
    warning_features: int
    critical_features: int


def population_stability_index(
    values: list[float], boundaries: list[float], expected_shares: list[float], *, epsilon: float = 1e-6
) -> float:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return 0.0
    bucket_ids = np.searchsorted(np.asarray(boundaries, dtype=float), numeric, side="right")
    actual_counts = np.bincount(bucket_ids, minlength=len(expected_shares)).astype(float)
    actual = (actual_counts + epsilon) / (actual_counts.sum() + epsilon * len(actual_counts))
    expected = np.asarray(expected_shares, dtype=float)
    expected = (expected + epsilon) / (expected.sum() + epsilon * len(expected))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def calculate_drift(
    feature_rows: list[dict[str, float]],
    reference: MonitoringReference,
    *,
    min_samples: int,
    warning_threshold: float,
    critical_threshold: float,
) -> DriftResult:
    if len(feature_rows) < min_samples:
        return DriftResult(False, "insufficient_data", {}, 0, 0)
    scores: dict[str, float] = {}
    for name in reference.feature_names:
        values = [float(row[name]) for row in feature_rows if name in row]
        item = reference.features[name]
        scores[name] = population_stability_index(
            values, item["boundaries"], item["expected_shares"]
        )
    critical = sum(score >= critical_threshold for score in scores.values())
    warning = sum(
        warning_threshold <= score < critical_threshold for score in scores.values()
    )
    status = "critical" if critical else "warning" if warning else "stable"
    return DriftResult(True, status, scores, warning, critical)
