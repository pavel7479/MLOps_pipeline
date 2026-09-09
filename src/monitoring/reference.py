"""Build and validate a compact Train-only drift reference artifact."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REFERENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MonitoringReference:
    metadata: dict[str, Any]
    feature_names: tuple[str, ...]
    features: dict[str, dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_reference(values: pd.Series, quantile_bins: int) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        raise ValueError(f"Cannot build drift reference for empty feature {values.name}")
    quantiles = np.linspace(0, 1, quantile_bins + 1)[1:-1]
    boundaries = np.unique(np.quantile(numeric, quantiles)).astype(float)
    boundaries = boundaries[(boundaries > numeric.min()) & (boundaries < numeric.max())]
    bins = np.searchsorted(boundaries, numeric, side="right")
    counts = np.bincount(bins, minlength=len(boundaries) + 1)
    shares = counts / counts.sum()
    return {
        "boundaries": boundaries.tolist(),
        "expected_shares": shares.astype(float).tolist(),
        "count": int(numeric.size),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
    }


def build_reference(
    train: pd.DataFrame,
    feature_names: tuple[str, ...],
    *,
    symbol: str,
    timeframe: str,
    train_path: Path,
    manifest_path: Path,
    quantile_bins: int = 10,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    missing = sorted(set(feature_names) - set(train.columns))
    if missing:
        raise ValueError(f"Train dataset is missing features: {missing}")
    if quantile_bins < 2:
        raise ValueError("quantile_bins must be at least 2")
    timestamp_column = "timestamp" if "timestamp" in train.columns else None
    if timestamp_column:
        timestamps = pd.to_datetime(train[timestamp_column], utc=True)
        train_start, train_end = timestamps.min().isoformat(), timestamps.max().isoformat()
    else:
        train_start = train_end = None
    timestamp = generated_at or datetime.now(timezone.utc)
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "train_start": train_start,
            "train_end": train_end,
            "train_rows": int(len(train)),
            "features_count": len(feature_names),
            "train_sha256": sha256_file(train_path),
            "manifest_sha256": sha256_file(manifest_path),
            "generated_at": timestamp.astimezone(timezone.utc).isoformat(),
            "quantile_bins": quantile_bins,
        },
        "feature_names": list(feature_names),
        "features": {
            name: _feature_reference(train[name], quantile_bins)
            for name in feature_names
        },
    }


def write_reference(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_reference(
    path: str | Path, expected_features: tuple[str, ...] | None = None
) -> MonitoringReference:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load monitoring reference {source}: {exc}") from exc
    names = tuple(payload.get("feature_names", ()))
    metadata = payload.get("metadata", {})
    features = payload.get("features", {})
    if payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported monitoring reference schema_version")
    if len(names) != 28 or metadata.get("features_count") != 28:
        raise ValueError("Monitoring reference must contain exactly 28 features")
    if expected_features is not None and names != expected_features:
        raise ValueError("Monitoring reference does not match the feature contract")
    if set(features) != set(names):
        raise ValueError("Monitoring reference feature payload is incomplete")
    for name in names:
        item = features[name]
        if len(item.get("expected_shares", [])) != len(item.get("boundaries", [])) + 1:
            raise ValueError(f"Invalid drift bins for feature {name}")
    return MonitoringReference(metadata=metadata, feature_names=names, features=features)
