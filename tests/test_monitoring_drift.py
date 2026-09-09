from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.drift import calculate_drift, population_stability_index
from src.monitoring.reference import MonitoringReference, build_reference


def _reference(values: list[float]) -> MonitoringReference:
    series = pd.Series(values, name="x")
    numeric = np.asarray(values)
    boundaries = np.unique(np.quantile(numeric, np.linspace(0, 1, 11)[1:-1]))
    boundaries = boundaries[(boundaries > numeric.min()) & (boundaries < numeric.max())]
    buckets = np.searchsorted(boundaries, numeric, side="right")
    expected = np.bincount(buckets, minlength=len(boundaries) + 1) / len(values)
    return MonitoringReference(
        metadata={}, feature_names=("x",),
        features={"x": {"boundaries": boundaries.tolist(), "expected_shares": expected.tolist()}},
    )


def test_psi_is_stable_for_same_distribution_and_critical_for_shift() -> None:
    reference = _reference(list(range(100)))
    stable = calculate_drift(
        [{"x": float(value)} for value in range(100)], reference,
        min_samples=100, warning_threshold=0.1, critical_threshold=0.25,
    )
    shifted = calculate_drift(
        [{"x": 1000.0 + value} for value in range(100)], reference,
        min_samples=100, warning_threshold=0.1, critical_threshold=0.25,
    )
    assert stable.status == "stable" and stable.scores["x"] < 0.1
    assert shifted.status == "critical" and shifted.critical_features == 1


def test_insufficient_data_does_not_claim_drift() -> None:
    result = calculate_drift(
        [{"x": 1.0}], _reference(list(range(20))),
        min_samples=2, warning_threshold=0.1, critical_threshold=0.25,
    )
    assert result.available is False
    assert result.status == "insufficient_data"
    assert result.scores == {}


def test_psi_handles_zero_bins_duplicates_and_out_of_range_values() -> None:
    score = population_stability_index(
        [-100.0, -100.0, 100.0, 100.0], [0.0, 1.0], [0.5, 0.0, 0.5]
    )
    assert np.isfinite(score) and score >= 0


def test_reference_meaningful_content_is_reproducible(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    manifest_path = tmp_path / "manifest.json"
    train_path.write_bytes(b"same-train")
    manifest_path.write_text("same-manifest", encoding="utf-8")
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=20, tz="UTC"), "x": range(20)})
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = build_reference(
        frame, ("x",), symbol="BTCUSDT", timeframe="1h",
        train_path=train_path, manifest_path=manifest_path, generated_at=fixed,
    )
    second = build_reference(
        frame, ("x",), symbol="BTCUSDT", timeframe="1h",
        train_path=train_path, manifest_path=manifest_path, generated_at=fixed,
    )
    assert first == second
    assert first["metadata"]["train_rows"] == 20
