"""SQLAlchemy 2.x ORM entities for inference history."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Float, Index, JSON, String, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PredictionRecord(Base):
    """One committed, traceable, idempotent model prediction."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "prediction IN ('BUY', 'HOLD', 'SELL')",
            name="ck_predictions_prediction",
        ),
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_prediction", "prediction"),
        Index("ix_predictions_model_version", "model_version"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    request_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), nullable=False, unique=True
    )
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    feature_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False
    )
    prediction: Mapped[str] = mapped_column(String(8), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
