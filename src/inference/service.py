"""Prediction business logic independent from FastAPI and MLflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID
from typing import TYPE_CHECKING

import numpy as np

from src.database import (
    DuplicatePredictionError,
    PredictionCreate,
    PredictionRecord,
    PredictionRepository,
)

from .feature_validator import FeatureValidator
from .model_provider import LoadedModel

if TYPE_CHECKING:
    from src.monitoring.metrics import ApplicationMetrics

VALID_PREDICTIONS = {"BUY", "HOLD", "SELL"}


class PredictionConflictError(RuntimeError):
    """One request_id cannot represent two different payloads."""


class InvalidModelPredictionError(RuntimeError):
    """The loaded model violated the public output contract."""


@dataclass(frozen=True)
class PredictionInput:
    request_id: UUID
    symbol: str
    timeframe: str
    feature_timestamp: datetime
    features: dict[str, int | float]


@dataclass(frozen=True)
class PredictionResult:
    request_id: UUID
    symbol: str
    timeframe: str
    feature_timestamp: datetime
    prediction: str
    model_name: str
    model_alias: str
    model_version: str
    model_run_id: str
    latency_ms: float
    replayed: bool
    created_at: datetime


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class PredictionService:
    """Validate, deduplicate, predict, commit, and return one result."""

    def __init__(
        self,
        loaded_model: LoadedModel,
        repository: PredictionRepository,
        validator: FeatureValidator,
        metrics: ApplicationMetrics | None = None,
    ) -> None:
        self.loaded_model = loaded_model
        self.repository = repository
        self.validator = validator
        self.metrics = metrics

    @staticmethod
    def _from_record(
        record: PredictionRecord, *, replayed: bool
    ) -> PredictionResult:
        return PredictionResult(
            request_id=record.request_id,
            symbol=record.symbol,
            timeframe=record.timeframe,
            feature_timestamp=_aware(record.feature_timestamp),
            prediction=record.prediction,
            model_name=record.model_name,
            model_alias=record.model_alias,
            model_version=record.model_version,
            model_run_id=record.model_run_id,
            latency_ms=record.latency_ms,
            replayed=replayed,
            created_at=_aware(record.created_at),
        )

    def predict(self, request: PredictionInput) -> PredictionResult:
        started = perf_counter()
        frame = self.validator.validate_and_prepare(
            symbol=request.symbol,
            timeframe=request.timeframe,
            feature_timestamp=request.feature_timestamp,
            features=request.features,
        )
        request_hash = self.validator.fingerprint(
            symbol=request.symbol,
            timeframe=request.timeframe,
            feature_timestamp=request.feature_timestamp,
            features=request.features,
        )
        existing = self.repository.get_by_request_id(request.request_id)
        if existing is not None:
            if existing.request_hash != request_hash:
                if self.metrics:
                    self.metrics.record_conflict()
                raise PredictionConflictError(
                    "request_id already exists with a different payload"
                )
            if self.metrics:
                self.metrics.record_replay()
            return self._from_record(existing, replayed=True)

        inference_started = perf_counter()
        try:
            values = np.asarray(self.loaded_model.model.predict(frame)).reshape(-1)
        finally:
            if self.metrics:
                self.metrics.observe_inference(
                    self.loaded_model.version, perf_counter() - inference_started
                )
        if len(values) != 1:
            raise InvalidModelPredictionError(
                "Model must return exactly one prediction"
            )
        prediction = str(values[0])
        if prediction not in VALID_PREDICTIONS:
            raise InvalidModelPredictionError(
                "Model returned a value outside BUY/HOLD/SELL"
            )
        latency_ms = (perf_counter() - started) * 1000
        create = PredictionCreate(
            request_id=request.request_id,
            request_hash=request_hash,
            feature_timestamp=request.feature_timestamp,
            symbol=request.symbol,
            timeframe=request.timeframe,
            features={
                name: float(frame.iloc[0][name])
                for name in self.loaded_model.feature_names
            },
            prediction=prediction,
            model_name=self.loaded_model.registered_model_name,
            model_alias=self.loaded_model.alias,
            model_version=self.loaded_model.version,
            model_run_id=self.loaded_model.run_id,
            latency_ms=latency_ms,
        )
        try:
            record = self.repository.create(create)
        except DuplicatePredictionError as exc:
            if exc.record.request_hash != request_hash:
                if self.metrics:
                    self.metrics.record_conflict()
                raise PredictionConflictError(
                    "request_id already exists with a different payload"
                ) from exc
            if self.metrics:
                self.metrics.record_replay()
            return self._from_record(exc.record, replayed=True)
        except Exception:
            if self.metrics:
                self.metrics.record_database_write_failure()
            raise
        if self.metrics:
            self.metrics.record_prediction(prediction, self.loaded_model.version)
        return self._from_record(record, replayed=False)
