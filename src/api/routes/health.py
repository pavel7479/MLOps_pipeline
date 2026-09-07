"""Cheap liveness and dependency-aware readiness endpoints."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from src.api.schemas import LivenessResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LivenessResponse)
def live() -> LivenessResponse:
    return LivenessResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
def ready(request: Request):
    model_ready = getattr(request.app.state, "loaded_model", None) is not None
    database_ready = request.app.state.database_health_checker(
        request.app.state.database.engine
    )
    payload = ReadinessResponse(
        status="ready" if model_ready and database_ready else "unavailable",
        model="ready" if model_ready else "unavailable",
        database="ready" if database_ready else "unavailable",
    )
    if payload.status != "ready":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(),
        )
    return payload
