"""Persistence boundary for prediction records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import PredictionRecord


@dataclass(frozen=True)
class PredictionCreate:
    request_id: UUID
    request_hash: str
    feature_timestamp: datetime
    symbol: str
    timeframe: str
    features: dict[str, float]
    prediction: str
    model_name: str
    model_alias: str
    model_version: str
    model_run_id: str
    latency_ms: float


class DuplicatePredictionError(RuntimeError):
    """Raised after a database uniqueness race, with the existing record."""

    def __init__(self, record: PredictionRecord) -> None:
        super().__init__(f"Prediction request already exists: {record.request_id}")
        self.record = record


class PredictionRepository:
    """SQLAlchemy expressions only; no user-built SQL strings."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, values: PredictionCreate) -> PredictionRecord:
        record = PredictionRecord(**vars(values))
        self.session.add(record)
        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            existing = self.get_by_request_id(values.request_id)
            if existing is not None:
                raise DuplicatePredictionError(existing) from exc
            raise
        self.session.refresh(record)
        return record

    def get_by_request_id(self, request_id: UUID) -> PredictionRecord | None:
        return self.session.scalar(
            select(PredictionRecord).where(
                PredictionRecord.request_id == request_id
            )
        )

    def list_recent(
        self,
        *,
        limit: int,
        offset: int,
        prediction: str | None = None,
    ) -> list[PredictionRecord]:
        statement = select(PredictionRecord)
        if prediction is not None:
            statement = statement.where(PredictionRecord.prediction == prediction)
        statement = statement.order_by(
            PredictionRecord.created_at.desc(), PredictionRecord.id.desc()
        ).limit(limit).offset(offset)
        return list(self.session.scalars(statement))

    def count(self, *, prediction: str | None = None) -> int:
        statement = select(func.count()).select_from(PredictionRecord)
        if prediction is not None:
            statement = statement.where(PredictionRecord.prediction == prediction)
        return int(self.session.scalar(statement) or 0)
