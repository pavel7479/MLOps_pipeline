"""Prediction creation and PostgreSQL history endpoints."""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError

from src.api.dependencies import get_prediction_service, get_repository
from src.api.schemas import (
    ModelReference,
    PredictionListResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.database import PredictionRecord, PredictionRepository
from src.inference import (
    FeatureValidationError,
    InvalidModelPredictionError,
    PredictionConflictError,
    PredictionInput,
    PredictionResult,
    PredictionService,
)

LOGGER = logging.getLogger(__name__)
router = APIRouter(tags=["predictions"])


def _response(result: PredictionResult) -> PredictionResponse:
    return PredictionResponse(
        request_id=result.request_id,
        symbol=result.symbol,
        timeframe=result.timeframe,
        feature_timestamp=result.feature_timestamp,
        prediction=result.prediction,
        model=ModelReference(
            registered_name=result.model_name,
            alias=result.model_alias,
            version=result.model_version,
            run_id=result.model_run_id,
        ),
        latency_ms=result.latency_ms,
        replayed=result.replayed,
        created_at=result.created_at,
    )


def _record_response(
    record: PredictionRecord, *, replayed: bool = False
) -> PredictionResponse:
    return _response(PredictionService._from_record(record, replayed=replayed))


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    try:
        result = service.predict(
            PredictionInput(
                request_id=payload.request_id,
                symbol=payload.symbol,
                timeframe=payload.timeframe,
                feature_timestamp=payload.feature_timestamp,
                features=dict(payload.features),
            )
        )
    except FeatureValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except PredictionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except SQLAlchemyError as exc:
        LOGGER.exception("Prediction database operation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction database is unavailable",
        ) from exc
    except InvalidModelPredictionError as exc:
        LOGGER.exception("Loaded model violated prediction contract")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction failed",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected inference failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        ) from exc
    LOGGER.info(
        "prediction request_id=%s symbol=%s timeframe=%s feature_timestamp=%s "
        "prediction=%s model_version=%s latency_ms=%.3f replayed=%s",
        result.request_id,
        result.symbol,
        result.timeframe,
        result.feature_timestamp.isoformat(),
        result.prediction,
        result.model_version,
        result.latency_ms,
        result.replayed,
    )
    return _response(result)


@router.get("/predictions", response_model=PredictionListResponse)
def list_predictions(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    prediction: Literal["BUY", "HOLD", "SELL"] | None = None,
    repository: PredictionRepository = Depends(get_repository),
) -> PredictionListResponse:
    try:
        records = repository.list_recent(
            limit=limit, offset=offset, prediction=prediction
        )
        total = repository.count(prediction=prediction)
    except SQLAlchemyError as exc:
        LOGGER.exception("Prediction history query failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction database is unavailable",
        ) from exc
    return PredictionListResponse(
        items=[_record_response(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/predictions/{request_id}", response_model=PredictionResponse)
def get_prediction(
    request_id: UUID,
    repository: PredictionRepository = Depends(get_repository),
) -> PredictionResponse:
    try:
        record = repository.get_by_request_id(request_id)
    except SQLAlchemyError as exc:
        LOGGER.exception("Prediction lookup failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction database is unavailable",
        ) from exc
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found"
        )
    return _record_response(record)
