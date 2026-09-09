"""Bounded, read-only database access for the monitoring worker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.database.models import PredictionRecord


@dataclass(frozen=True)
class MonitoringPrediction:
    features: dict[str, float]
    prediction: str
    created_at: datetime


class MonitoringRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def fetch_recent(self, limit: int) -> list[MonitoringPrediction]:
        if limit <= 0:
            raise ValueError("monitoring query limit must be positive")
        statement = (
            select(
                PredictionRecord.features,
                PredictionRecord.prediction,
                PredictionRecord.created_at,
            )
            .order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
            .limit(limit)
        )
        with self.session_factory() as session:
            rows = session.execute(statement).all()
        return [
            MonitoringPrediction(
                features={key: float(value) for key, value in row.features.items()},
                prediction=str(row.prediction),
                created_at=row.created_at,
            )
            for row in rows
        ]
