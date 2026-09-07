"""Request-scoped dependencies backed by application state."""

from collections.abc import Generator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.database import PredictionRepository
from src.inference import LoadedModel, PredictionService


def get_session(request: Request) -> Generator[Session, None, None]:
    session = request.app.state.database.session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_repository(
    session: Session = Depends(get_session),
) -> PredictionRepository:
    return PredictionRepository(session)


def get_loaded_model(request: Request) -> LoadedModel:
    loaded = getattr(request.app.state, "loaded_model", None)
    if loaded is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable",
        )
    return loaded


def get_prediction_service(
    request: Request,
    repository: PredictionRepository = Depends(get_repository),
    loaded_model: LoadedModel = Depends(get_loaded_model),
) -> PredictionService:
    return PredictionService(
        loaded_model=loaded_model,
        repository=repository,
        validator=request.app.state.feature_validator,
    )
