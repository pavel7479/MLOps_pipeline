# syntax=docker/dockerfile:1.7

FROM python:3.12.4-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

FROM base AS builder

COPY requirements-api.txt .
RUN python -m pip wheel --wheel-dir /wheels -r requirements-api.txt

FROM builder AS test-builder

COPY requirements-test.txt .
RUN mkdir /test-wheels \
    && cp /wheels/* /test-wheels/ \
    && python -m pip wheel --wheel-dir /test-wheels \
        pytest==8.4.2 pyarrow==19.0.1

FROM base AS runtime

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home --home-dir /home/appuser appuser

COPY requirements-api.txt .
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    python -m pip install --no-index --find-links=/wheels -r requirements-api.txt \
    && python -m pip check

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser config.yaml alembic.ini ./
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser scripts/init_mlflow_registry.py ./scripts/init_mlflow_registry.py

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM runtime AS test

USER root
ENV PYTEST_ADDOPTS="-p no:cacheprovider"
COPY requirements-test.txt .
RUN --mount=type=bind,from=test-builder,source=/test-wheels,target=/test-wheels \
    python -m pip install --no-index --find-links=/test-wheels -r requirements-test.txt \
    && python -m pip check

COPY --chown=appuser:appuser tests ./tests
COPY --chown=appuser:appuser scripts/smoke_test_api.py ./scripts/smoke_test_api.py
COPY --chown=appuser:appuser Dockerfile compose.yaml .dockerignore ./

USER appuser

CMD ["python", "-m", "pytest", "-v"]
