from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from prometheus_client import CollectorRegistry, generate_latest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.database import Base, PredictionCreate, PredictionRepository
from src.monitoring.metrics import WorkerMetrics
from src.monitoring.reference import MonitoringReference
from src.monitoring.repository import MonitoringPrediction, MonitoringRepository
from src.monitoring.worker import MonitoringWorker


class FlakyRepository:
    def __init__(self, rows):
        self.rows = rows
        self.fail = True
        self.limits = []

    def fetch_recent(self, limit):
        self.limits.append(limit)
        if self.fail:
            raise RuntimeError("db down")
        return self.rows[:limit]


def _settings():
    return SimpleNamespace(
        monitoring=SimpleNamespace(
            window_size=500, min_samples=2,
            warning_threshold=0.1, critical_threshold=0.25,
        ),
        mlflow=SimpleNamespace(tracking_uri="http://mlflow:5000"),
    )


def _reference():
    return MonitoringReference(
        metadata={}, feature_names=("x",),
        features={"x": {"boundaries": [0.0], "expected_shares": [0.5, 0.5]}},
    )


def test_worker_survives_database_failure_and_records_recovery() -> None:
    rows = [
        MonitoringPrediction({"x": -1.0}, "BUY", datetime.now(timezone.utc)),
        MonitoringPrediction({"x": 1.0}, "SELL", datetime.now(timezone.utc)),
    ]
    repository = FlakyRepository(rows)
    metrics = WorkerMetrics(("x",), CollectorRegistry())
    worker = MonitoringWorker(
        repository, _reference(), metrics, _settings(),
        mlflow_check=lambda _uri: True, clock=lambda: 123.0,
    )
    assert worker.run_cycle() is False
    repository.fail = False
    assert worker.run_cycle() is True
    text = generate_latest(metrics.registry).decode()
    assert "crypto_database_reachable 1.0" in text
    assert "crypto_monitoring_window_samples 2.0" in text
    assert 'crypto_recent_prediction_share{prediction="BUY"} 0.5' in text
    assert 'crypto_recent_prediction_share{prediction="HOLD"} 0.0' in text
    assert 'crypto_recent_prediction_share{prediction="SELL"} 0.5' in text
    assert "crypto_last_drift_check_timestamp_seconds 123.0" in text
    assert repository.limits == [500, 500]


def test_mlflow_failure_does_not_block_drift_cycle() -> None:
    repository = FlakyRepository([])
    repository.fail = False
    metrics = WorkerMetrics(("x",), CollectorRegistry())
    worker = MonitoringWorker(
        repository, _reference(), metrics, _settings(), mlflow_check=lambda _uri: False
    )
    assert worker.run_cycle() is True
    text = generate_latest(metrics.registry).decode()
    assert "crypto_mlflow_reachable 0.0" in text
    assert "crypto_database_reachable 1.0" in text


def test_monitoring_repository_applies_database_limit_and_returns_only_contract() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    with factory() as session:
        repository = PredictionRepository(session)
        for index in range(6):
            repository.create(PredictionCreate(
                request_id=uuid4(), request_hash=f"{index:064d}",
                feature_timestamp=datetime.now(timezone.utc), symbol="BTCUSDT",
                timeframe="1h", features={"x": float(index)}, prediction="HOLD",
                model_name="model", model_alias="champion", model_version="1",
                model_run_id="run", latency_ms=1.0,
            ))
    rows = MonitoringRepository(factory).fetch_recent(limit=2)
    assert len(rows) == 2
    assert all(set(vars(row)) == {"features", "prediction", "created_at"} for row in rows)
    engine.dispose()
