"""FastAPI application factory for local model inference."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from sqlalchemy.engine import Engine

from src.config import AppSettings, load_settings
from src.database import DatabaseRuntime, create_database, database_is_ready
from src.inference import (
    FeatureValidator,
    MLflowModelProvider,
    ModelProvider,
)

from .routes import health_router, model_router, predictions_router

LOGGER = logging.getLogger(__name__)


def create_app(
    settings: AppSettings | None = None,
    *,
    database: DatabaseRuntime | None = None,
    model_provider: ModelProvider | None = None,
    database_health_checker: Callable[[Engine], bool] = database_is_ready,
    allow_model_load_failure: bool = False,
) -> FastAPI:
    application_settings = settings or load_settings()
    supplied_database = database
    supplied_provider = model_provider

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime_database = supplied_database or create_database(
            application_settings.database
        )
        owns_database = supplied_database is None
        provider = supplied_provider or MLflowModelProvider(
            application_settings.mlflow, application_settings.inference
        )
        application.state.settings = application_settings
        application.state.database = runtime_database
        application.state.database_health_checker = database_health_checker
        application.state.loaded_model = None
        if not database_health_checker(runtime_database.engine):
            LOGGER.warning(
                "PostgreSQL is unavailable at startup; readiness will remain false"
            )
        try:
            loaded = provider.load_model()
        except Exception:
            if not allow_model_load_failure:
                if owns_database:
                    runtime_database.dispose()
                raise
            LOGGER.exception("Model unavailable during test/degraded startup")
        else:
            application.state.loaded_model = loaded
            application.state.feature_validator = FeatureValidator(
                loaded.feature_names,
                application_settings.inference.expected_symbol,
                application_settings.inference.expected_timeframe,
            )
            LOGGER.info(
                "Loaded model %s@%s version=%s run_id=%s features=%d",
                loaded.registered_model_name,
                loaded.alias,
                loaded.version,
                loaded.run_id,
                len(loaded.feature_names),
            )
        try:
            yield
        finally:
            if owns_database:
                runtime_database.dispose()

    application = FastAPI(
        title=application_settings.api.title,
        version=application_settings.api.version,
        lifespan=lifespan,
    )
    prefix = "/api/v1"
    application.include_router(health_router, prefix=prefix)
    application.include_router(model_router, prefix=prefix)
    application.include_router(predictions_router, prefix=prefix)
    return application


app = create_app()
