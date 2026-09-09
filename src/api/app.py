"""FastAPI application factory for local model inference."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.routing import compile_path
from sqlalchemy.engine import Engine

from src.config import AppSettings, load_settings
from src.database import DatabaseRuntime, create_database, database_is_ready
from src.inference import (
    FeatureValidator,
    MLflowModelProvider,
    ModelProvider,
)
from src.monitoring.metrics import ApplicationMetrics

from .routes import health_router, model_router, predictions_router

LOGGER = logging.getLogger(__name__)


def create_app(
    settings: AppSettings | None = None,
    *,
    database: DatabaseRuntime | None = None,
    model_provider: ModelProvider | None = None,
    database_health_checker: Callable[[Engine], bool] = database_is_ready,
    allow_model_load_failure: bool = False,
    metrics: ApplicationMetrics | None = None,
) -> FastAPI:
    application_settings = settings or load_settings()
    supplied_database = database
    supplied_provider = model_provider
    application_metrics = metrics or ApplicationMetrics()

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
            application_metrics.set_loaded_model(
                loaded.registered_model_name,
                loaded.alias,
                loaded.version,
                loaded.backtest_status,
            )
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
    application.state.metrics = application_metrics

    @application.middleware("http")
    async def observe_http(request, call_next):
        started = perf_counter()
        status_code = 500
        route_template = "unmatched"
        candidates = []
        for route in application.routes:
            original_router = getattr(route, "original_router", None)
            if original_router is None:
                candidates.append((getattr(route, "path", None), route))
                continue
            prefix = route.include_context.prefix
            candidates.extend(
                (f"{prefix}{nested.path}", nested)
                for nested in original_router.routes
                if getattr(nested, "path", None) is not None
            )
        for path, route in candidates:
            if path is None:
                continue
            path_regex = compile_path(path)[0]
            methods = getattr(route, "methods", None)
            if path_regex.fullmatch(request.scope["path"]) and (
                methods is None or request.method in methods
            ):
                route_template = path
                break
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            method = request.method
            application_metrics.http_requests.labels(
                method=method,
                route=route_template,
                status_code=str(status_code),
            ).inc()
            application_metrics.http_duration.labels(
                method=method, route=route_template
            ).observe(perf_counter() - started)

    @application.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(application_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    prefix = "/api/v1"
    application.include_router(health_router, prefix=prefix)
    application.include_router(model_router, prefix=prefix)
    application.include_router(predictions_router, prefix=prefix)
    return application


app = create_app()
