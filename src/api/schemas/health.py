"""Health response models."""

from typing import Literal

from pydantic import BaseModel


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["crypto_ml_inference"] = "crypto_ml_inference"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    model: Literal["ready", "unavailable"]
    database: Literal["ready", "unavailable"]
