from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import create_app
from src.config import APISettings, DatabaseSettings, InferenceSettings
from src.config.settings import (
    AppSettings,
    HttpSettings,
    LoggingSettings,
    MarketDataSettings,
    StorageSettings,
)
from src.database import Base, DatabaseRuntime
from src.inference import LoadedModel

FEATURE_NAMES = tuple(f"feature_{index:02d}" for index in range(28))


class FakeModel:
    def __init__(self, label: str = "BUY") -> None:
        self.label = label
        self.calls = 0

    def predict(self, frame: pd.DataFrame):
        self.calls += 1
        assert list(frame.columns) == list(FEATURE_NAMES)
        return np.asarray([self.label])


class FakeProvider:
    def __init__(
        self,
        model: FakeModel | None = None,
        *,
        fail: bool = False,
        version: str = "1",
    ) -> None:
        self.model = model or FakeModel()
        self.fail = fail
        self.version = version
        self.load_calls = 0

    def load_model(self) -> LoadedModel:
        self.load_calls += 1
        if self.fail:
            raise RuntimeError("model load failed")
        return LoadedModel(
            model=self.model,
            registered_model_name="crypto_direction_classifier",
            alias="champion",
            version=self.version,
            run_id=f"run-id-{self.version}",
            feature_names=FEATURE_NAMES,
            loaded_at=datetime(2026, 9, 7, 12, tzinfo=timezone.utc),
            backtest_status="failed_profitability_check",
        )


def _settings(tmp_path) -> AppSettings:
    return AppSettings(
        market_data=MarketDataSettings(
            "binance", "BTCUSDT", "1h",
            datetime(2024, 1, 1, tzinfo=timezone.utc), None,
        ),
        storage=StorageSettings(tmp_path / "raw", tmp_path / "processed", "parquet"),
        http=HttpSettings(1, 0, 0),
        logging=LoggingSettings("INFO"),
        api=APISettings(),
        inference=InferenceSettings(),
        database=DatabaseSettings(),
    )


def _database() -> DatabaseRuntime:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return DatabaseRuntime(
        engine=engine,
        session_factory=sessionmaker(
            bind=engine, class_=Session, expire_on_commit=False
        ),
    )


def _payload(request_id=None) -> dict:
    return {
        "request_id": str(request_id or uuid4()),
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "feature_timestamp": "2026-09-07T12:00:00Z",
        "features": {
            name: float(index) / 100 for index, name in enumerate(FEATURE_NAMES)
        },
    }


def test_health_model_swagger_and_model_loaded_once(tmp_path) -> None:
    database = _database()
    provider = FakeProvider()
    app = create_app(
        _settings(tmp_path), database=database, model_provider=provider
    )
    with TestClient(app) as client:
        assert client.get("/api/v1/health/live").json() == {
            "status": "ok", "service": "crypto_ml_inference"
        }
        assert client.get("/api/v1/health/ready").json() == {
            "status": "ready", "model": "ready", "database": "ready"
        }
        model = client.get("/api/v1/model")
        assert model.status_code == 200
        assert model.json()["features_count"] == 28
        assert model.json()["version"] == "1"
        assert client.get("/docs").status_code == 200
        paths = client.get("/openapi.json").json()["paths"]
        assert {
            "/api/v1/health/live",
            "/api/v1/health/ready",
            "/api/v1/model",
            "/api/v1/predict",
            "/api/v1/predictions",
            "/api/v1/predictions/{request_id}",
        } <= set(paths)
        client.get("/api/v1/model")
        client.get("/api/v1/health/ready")
    assert provider.load_calls == 1
    database.dispose()


def test_prediction_persistence_idempotency_conflict_and_history(tmp_path) -> None:
    database = _database()
    provider = FakeProvider(FakeModel("HOLD"))
    app = create_app(
        _settings(tmp_path), database=database, model_provider=provider
    )
    payload = _payload()
    with TestClient(app) as client:
        first = client.post("/api/v1/predict", json=payload)
        replay = client.post("/api/v1/predict", json=payload)
        lookup = client.get(f"/api/v1/predictions/{payload['request_id']}")
        history = client.get("/api/v1/predictions?limit=1&offset=0")
        changed = {
            **payload,
            "features": {**payload["features"], FEATURE_NAMES[0]: 99.0},
        }
        conflict = client.post("/api/v1/predict", json=changed)
    assert first.status_code == replay.status_code == lookup.status_code == 200
    assert first.json()["replayed"] is False
    assert replay.json()["replayed"] is True
    assert lookup.json()["prediction"] == "HOLD"
    assert lookup.json()["model"]["version"] == "1"
    assert history.json()["total"] == 1
    assert len(history.json()["items"]) == 1
    assert conflict.status_code == 409
    assert provider.model.calls == 1
    with database.session_factory() as session:
        count = session.scalar(
            select(func.count()).select_from(Base.metadata.tables["predictions"])
        )
        assert count == 1
    database.dispose()


def test_restart_loads_new_champion_but_replay_keeps_old_version(tmp_path) -> None:
    database = _database()
    payload = _payload()
    first_provider = FakeProvider(FakeModel("BUY"), version="1")
    first_app = create_app(
        _settings(tmp_path), database=database, model_provider=first_provider
    )
    with TestClient(first_app) as client:
        first = client.post("/api/v1/predict", json=payload)
    assert first.status_code == 200
    assert first.json()["model"]["version"] == "1"

    replacement_provider = FakeProvider(FakeModel("SELL"), version="3")
    replacement_app = create_app(
        _settings(tmp_path), database=database, model_provider=replacement_provider
    )
    with TestClient(replacement_app) as client:
        replay = client.post("/api/v1/predict", json=payload)
        fresh_payload = {**payload, "request_id": str(uuid4())}
        fresh = client.post("/api/v1/predict", json=fresh_payload)
    assert replay.status_code == fresh.status_code == 200
    assert replay.json()["replayed"] is True
    assert replay.json()["prediction"] == "BUY"
    assert replay.json()["model"]["version"] == "1"
    assert fresh.json()["prediction"] == "SELL"
    assert fresh.json()["model"]["version"] == "3"
    assert replacement_provider.load_calls == 1
    assert replacement_provider.model.calls == 1
    database.dispose()


def test_api_generates_request_id_when_client_omits_it(tmp_path) -> None:
    database = _database()
    app = create_app(
        _settings(tmp_path), database=database, model_provider=FakeProvider()
    )
    payload = _payload()
    payload.pop("request_id")
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert UUID(response.json()["request_id"])
    database.dispose()


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        (lambda body: body.update(symbol="ETHUSDT"), "symbol must be BTCUSDT"),
        (lambda body: body.update(timeframe="5m"), "timeframe must be 1h"),
        (
            lambda body: body["features"].pop(FEATURE_NAMES[0]),
            "missing features",
        ),
        (
            lambda body: body["features"].update(future_return=0.1),
            "unexpected features",
        ),
    ],
)
def test_api_rejects_domain_contract_violations(
    tmp_path, mutation, expected_detail
) -> None:
    database = _database()
    app = create_app(
        _settings(tmp_path), database=database, model_provider=FakeProvider()
    )
    payload = _payload()
    mutation(payload)
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    assert expected_detail in response.json()["detail"]
    database.dispose()


def test_api_rejects_naive_timestamp_and_invalid_pagination(tmp_path) -> None:
    database = _database()
    app = create_app(
        _settings(tmp_path), database=database, model_provider=FakeProvider()
    )
    payload = _payload()
    payload["feature_timestamp"] = "2026-09-07T12:00:00"
    with TestClient(app) as client:
        assert client.post("/api/v1/predict", json=payload).status_code == 422
        assert client.get("/api/v1/predictions?limit=101").status_code == 422
        assert client.get(f"/api/v1/predictions/{uuid4()}").status_code == 404
    database.dispose()


def test_readiness_reports_database_unavailable_without_secret(tmp_path) -> None:
    database = _database()
    app = create_app(
        _settings(tmp_path),
        database=database,
        model_provider=FakeProvider(),
        database_health_checker=lambda engine: False,
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")
        assert client.get("/api/v1/health/live").status_code == 200
    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable", "model": "ready", "database": "unavailable"
    }
    assert "postgresql" not in response.text.lower()
    database.dispose()


def test_readiness_reports_model_unavailable_in_degraded_test_mode(tmp_path) -> None:
    database = _database()
    app = create_app(
        _settings(tmp_path),
        database=database,
        model_provider=FakeProvider(fail=True),
        allow_model_load_failure=True,
    )
    with TestClient(app) as client:
        readiness = client.get("/api/v1/health/ready")
        model = client.get("/api/v1/model")
    assert readiness.status_code == 503
    assert readiness.json()["model"] == "unavailable"
    assert model.status_code == 503
    database.dispose()


def test_model_load_failure_aborts_normal_startup(tmp_path) -> None:
    database = _database()
    app = create_app(
        _settings(tmp_path),
        database=database,
        model_provider=FakeProvider(fail=True),
    )
    with pytest.raises(RuntimeError, match="model load failed"):
        with TestClient(app):
            pass
    database.dispose()


def test_metrics_endpoint_is_hidden_and_uses_route_templates(tmp_path) -> None:
    database = _database()
    app = create_app(
        _settings(tmp_path), database=database, model_provider=FakeProvider()
    )
    first_id, second_id = uuid4(), uuid4()
    with TestClient(app) as client:
        client.get(f"/api/v1/predictions/{first_id}")
        client.get(f"/api/v1/predictions/{second_id}")
        metrics = client.get("/metrics")
        paths = client.get("/openapi.json").json()["paths"]
    assert metrics.status_code == 200
    assert "/metrics" not in paths
    assert 'route="/api/v1/predictions/{request_id}"' in metrics.text
    assert str(first_id) not in metrics.text and str(second_id) not in metrics.text
    assert "crypto_loaded_model_info" in metrics.text
    assert "run-id-1" not in metrics.text
    database.dispose()


def test_metrics_endpoint_remains_available_in_degraded_mode(tmp_path) -> None:
    database = _database()
    app = create_app(
        _settings(tmp_path), database=database, model_provider=FakeProvider(fail=True),
        database_health_checker=lambda _engine: False,
        allow_model_load_failure=True,
    )
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "crypto_model_loaded 0.0" in response.text
    database.dispose()
