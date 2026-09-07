"""Strict inference contract and deterministic idempotency fingerprint."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from typing import Mapping

import pandas as pd


class FeatureValidationError(ValueError):
    """Safe domain validation error suitable for an HTTP 422 response."""


class FeatureValidator:
    def __init__(
        self,
        feature_names: tuple[str, ...],
        expected_symbol: str,
        expected_timeframe: str,
    ) -> None:
        if not feature_names:
            raise ValueError("Expected feature list cannot be empty")
        self.feature_names = feature_names
        self.expected_symbol = expected_symbol
        self.expected_timeframe = expected_timeframe

    def validate_and_prepare(
        self,
        *,
        symbol: str,
        timeframe: str,
        feature_timestamp: datetime,
        features: Mapping[str, int | float],
    ) -> pd.DataFrame:
        if symbol != self.expected_symbol:
            raise FeatureValidationError(
                f"symbol must be {self.expected_symbol}, got {symbol}"
            )
        if timeframe != self.expected_timeframe:
            raise FeatureValidationError(
                f"timeframe must be {self.expected_timeframe}, got {timeframe}"
            )
        if (
            feature_timestamp.tzinfo is None
            or feature_timestamp.utcoffset() is None
        ):
            raise FeatureValidationError("feature_timestamp must include timezone")
        expected = set(self.feature_names)
        actual = set(features)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing features: {missing}")
            if extra:
                details.append(f"unexpected features: {extra}")
            raise FeatureValidationError("; ".join(details))
        ordered: dict[str, float] = {}
        for name in self.feature_names:
            value = features[name]
            if isinstance(value, bool) or type(value) not in (int, float):
                raise FeatureValidationError(f"feature {name} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise FeatureValidationError(f"feature {name} must be finite")
            ordered[name] = numeric
        return pd.DataFrame([ordered], columns=self.feature_names)

    @staticmethod
    def fingerprint(
        *,
        symbol: str,
        timeframe: str,
        feature_timestamp: datetime,
        features: Mapping[str, int | float],
    ) -> str:
        timestamp = feature_timestamp.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        normalized = {
            "symbol": symbol,
            "timeframe": timeframe,
            "feature_timestamp": timestamp,
            "features": {key: float(value) for key, value in features.items()},
        }
        serialized = json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return sha256(serialized.encode("utf-8")).hexdigest()
