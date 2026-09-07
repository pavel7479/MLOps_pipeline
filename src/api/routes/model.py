"""Metadata for the in-memory Registry model."""

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_loaded_model
from src.api.schemas import ModelInfoResponse
from src.inference import LoadedModel

router = APIRouter(tags=["model"])


@router.get("/model", response_model=ModelInfoResponse)
def model_info(
    request: Request,
    loaded: LoadedModel = Depends(get_loaded_model),
) -> ModelInfoResponse:
    settings = request.app.state.settings.inference
    return ModelInfoResponse(
        registered_name=loaded.registered_model_name,
        alias=loaded.alias,
        version=loaded.version,
        run_id=loaded.run_id,
        features_count=len(loaded.feature_names),
        symbol=settings.expected_symbol,
        timeframe=settings.expected_timeframe,
        loaded_at=loaded.loaded_at,
        backtest_status=loaded.backtest_status,
    )
