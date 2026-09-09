from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from src.inference import (
    FeatureValidator,
    LoadedModel,
    PredictionConflictError,
    PredictionInput,
    PredictionService,
)
from src.monitoring.metrics import ApplicationMetrics


class Model:
    def predict(self, frame: pd.DataFrame):
        return np.asarray(["BUY"])


class Record:
    def __init__(self, values):
        self.__dict__.update(vars(values))
        self.created_at = datetime.now(timezone.utc)


class Repository:
    def __init__(self, *, fail_write: bool = False):
        self.rows = {}
        self.fail_write = fail_write

    def get_by_request_id(self, request_id):
        return self.rows.get(request_id)

    def create(self, values):
        if self.fail_write:
            raise RuntimeError("database unavailable")
        record = Record(values)
        self.rows[values.request_id] = record
        return record


def _service(repository, metrics):
    loaded = LoadedModel(
        model=Model(), registered_model_name="model", alias="champion",
        version="7", run_id="secret-run-id", feature_names=("a", "b"),
        loaded_at=datetime.now(timezone.utc),
        backtest_status="failed_profitability_check",
    )
    return PredictionService(
        loaded, repository, FeatureValidator(("a", "b"), "BTCUSDT", "1h"), metrics
    )


def _input(request_id=None, a=1.0):
    return PredictionInput(
        request_id=request_id or uuid4(), symbol="BTCUSDT", timeframe="1h",
        feature_timestamp=datetime(2026, 9, 9, 12, tzinfo=timezone.utc),
        features={"a": a, "b": 2.0},
    )


def test_prediction_metrics_distinguish_commit_replay_conflict_and_inference() -> None:
    metrics = ApplicationMetrics(CollectorRegistry())
    repository = Repository()
    service = _service(repository, metrics)
    request_id = uuid4()
    service.predict(_input(request_id))
    service.predict(_input(request_id))
    with pytest.raises(PredictionConflictError):
        service.predict(_input(request_id, a=3.0))
    text = generate_latest(metrics.registry).decode()
    assert 'crypto_ml_predictions_total{model_version="7",prediction="BUY"} 1.0' in text
    assert "crypto_ml_prediction_replays_total 1.0" in text
    assert "crypto_ml_prediction_conflicts_total 1.0" in text
    assert 'crypto_ml_inference_duration_seconds_count{model_version="7"} 1.0' in text


def test_database_write_failure_is_counted_without_false_prediction() -> None:
    metrics = ApplicationMetrics(CollectorRegistry())
    with pytest.raises(RuntimeError, match="database unavailable"):
        _service(Repository(fail_write=True), metrics).predict(_input())
    text = generate_latest(metrics.registry).decode()
    assert "crypto_database_write_failures_total 1.0" in text
    assert "crypto_ml_predictions_total{" not in text


def test_loaded_model_info_has_bounded_labels_and_no_run_id() -> None:
    metrics = ApplicationMetrics(CollectorRegistry())
    metrics.set_loaded_model("model", "champion", "7", "failed_profitability_check")
    text = generate_latest(metrics.registry).decode()
    assert "crypto_model_loaded 1.0" in text
    assert "failed_profitability_check" in text
    assert "run_id" not in text and "secret-run-id" not in text


def test_inference_histogram_times_only_model_predict(monkeypatch) -> None:
    ticks = iter([0.0, 10.0, 10.25, 20.0])
    monkeypatch.setattr("src.inference.service.perf_counter", lambda: next(ticks))
    metrics = ApplicationMetrics(CollectorRegistry())
    _service(Repository(), metrics).predict(_input())
    text = generate_latest(metrics.registry).decode()
    assert 'crypto_ml_inference_duration_seconds_sum{model_version="7"} 0.25' in text
