"""Library-independent contracts for loading and using prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import pandas as pd


class PredictionModel(Protocol):
    def predict(self, model_input: pd.DataFrame) -> Any:
        """Return one BUY/HOLD/SELL label per input row."""


@dataclass(frozen=True)
class LoadedModel:
    model: PredictionModel
    registered_model_name: str
    alias: str
    version: str
    run_id: str
    feature_names: tuple[str, ...]
    loaded_at: datetime
    backtest_status: str | None = None


class ModelProvider(Protocol):
    def load_model(self) -> LoadedModel:
        """Resolve and load one model during application startup."""
