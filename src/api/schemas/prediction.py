"""Typed request and response contracts for prediction endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt

FeatureFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
FeatureValue = StrictInt | FeatureFloat
PredictionLabel = Literal["BUY", "HOLD", "SELL"]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    symbol: str = Field(min_length=1, max_length=32)
    timeframe: str = Field(min_length=1, max_length=16)
    feature_timestamp: AwareDatetime
    features: dict[str, FeatureValue]


class ModelReference(BaseModel):
    registered_name: str
    alias: str
    version: str
    run_id: str


class PredictionResponse(BaseModel):
    request_id: UUID
    symbol: str
    timeframe: str
    feature_timestamp: datetime
    prediction: PredictionLabel
    model: ModelReference
    latency_ms: float
    replayed: bool
    created_at: datetime


class PredictionListResponse(BaseModel):
    items: list[PredictionResponse]
    total: int
    limit: int
    offset: int
