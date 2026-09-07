from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.database import (
    Base,
    DuplicatePredictionError,
    PredictionCreate,
    PredictionRepository,
)
from src.inference import (
    FeatureValidator,
    InvalidModelPredictionError,
    LoadedModel,
    PredictionConflictError,
    PredictionInput,
    PredictionService,
)

FEATURE_NAMES = ("feature_a", "feature_b")


class FakeModel:
    def __init__(self, prediction: str) -> None:
        self.prediction = prediction
        self.calls = 0
        self.columns: list[str] = []

    def predict(self, frame: pd.DataFrame):
        self.calls += 1
        self.columns = list(frame.columns)
        return np.asarray([self.prediction])


def _loaded(model: FakeModel, version: str = "1") -> LoadedModel:
    return LoadedModel(
        model=model,
        registered_model_name="crypto_direction_classifier",
        alias="champion",
        version=version,
        run_id=f"run-{version}",
        feature_names=FEATURE_NAMES,
        loaded_at=datetime.now(timezone.utc),
        backtest_status="failed_profitability_check",
    )


@pytest.fixture
def session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as database_session:
        yield database_session
    engine.dispose()


def _input(request_id=None, value: float = 1.0) -> PredictionInput:
    return PredictionInput(
        request_id=request_id or uuid4(),
        symbol="BTCUSDT",
        timeframe="1h",
        feature_timestamp=datetime(2026, 9, 7, 12, tzinfo=timezone.utc),
        features={"feature_b": 2.0, "feature_a": value},
    )


def _service(session: Session, model: FakeModel, version: str = "1"):
    return PredictionService(
        _loaded(model, version),
        PredictionRepository(session),
        FeatureValidator(FEATURE_NAMES, "BTCUSDT", "1h"),
    )


def test_service_predicts_in_manifest_order_and_persists(session: Session) -> None:
    model = FakeModel("BUY")
    request = _input()
    result = _service(session, model).predict(request)
    stored = PredictionRepository(session).get_by_request_id(request.request_id)
    assert result.prediction == "BUY"
    assert result.replayed is False
    assert model.columns == list(FEATURE_NAMES)
    assert stored is not None and stored.model_version == "1"
    assert stored.features == {"feature_a": 1.0, "feature_b": 2.0}


def test_idempotent_replay_skips_second_prediction(session: Session) -> None:
    model = FakeModel("HOLD")
    service = _service(session, model)
    request = _input()
    first = service.predict(request)
    replay = service.predict(request)
    assert first.replayed is False and replay.replayed is True
    assert first.prediction == replay.prediction
    assert model.calls == 1
    assert PredictionRepository(session).count() == 1


def test_same_request_id_with_changed_payload_conflicts(session: Session) -> None:
    model = FakeModel("SELL")
    service = _service(session, model)
    request_id = uuid4()
    service.predict(_input(request_id, value=1.0))
    with pytest.raises(PredictionConflictError):
        service.predict(_input(request_id, value=1.1))
    assert model.calls == 1


def test_replay_preserves_old_model_after_champion_change(session: Session) -> None:
    request = _input()
    first_model = FakeModel("BUY")
    first = _service(session, first_model, version="1").predict(request)
    replacement_model = FakeModel("SELL")
    replay = _service(session, replacement_model, version="3").predict(request)
    assert first.prediction == replay.prediction == "BUY"
    assert replay.model_version == "1"
    assert replay.replayed is True
    assert replacement_model.calls == 0


@pytest.mark.parametrize("label", ["BUY", "SELL"])
def test_service_accepts_replaceable_models_without_library_logic(
    session: Session, label: str
) -> None:
    result = _service(session, FakeModel(label)).predict(_input())
    assert result.prediction == label


def test_invalid_model_output_is_rejected(session: Session) -> None:
    with pytest.raises(InvalidModelPredictionError):
        _service(session, FakeModel("UNKNOWN")).predict(_input())
    assert PredictionRepository(session).count() == 0


def _create_values(request_id, created_offset: int, prediction: str):
    timestamp = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    return PredictionCreate(
        request_id=request_id,
        request_hash=f"{created_offset:064d}",
        feature_timestamp=timestamp + timedelta(hours=created_offset),
        symbol="BTCUSDT",
        timeframe="1h",
        features={"feature_a": float(created_offset)},
        prediction=prediction,
        model_name="model",
        model_alias="champion",
        model_version="1",
        model_run_id="run",
        latency_ms=1.0,
    )


def test_repository_get_list_pagination_filter_and_unique(session: Session) -> None:
    repository = PredictionRepository(session)
    first_id, second_id, third_id = uuid4(), uuid4(), uuid4()
    first = repository.create(_create_values(first_id, 1, "BUY"))
    second = repository.create(_create_values(second_id, 2, "SELL"))
    third = repository.create(_create_values(third_id, 3, "BUY"))
    first.created_at = datetime.now(timezone.utc) - timedelta(seconds=3)
    second.created_at = datetime.now(timezone.utc) - timedelta(seconds=2)
    third.created_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    session.commit()
    assert repository.get_by_request_id(second_id).prediction == "SELL"
    assert [row.request_id for row in repository.list_recent(limit=2, offset=0)] == [
        third_id, second_id
    ]
    assert [row.request_id for row in repository.list_recent(limit=2, offset=1)] == [
        second_id, first_id
    ]
    assert repository.count(prediction="BUY") == 2
    assert all(
        row.prediction == "BUY"
        for row in repository.list_recent(limit=10, offset=0, prediction="BUY")
    )
    with pytest.raises(DuplicatePredictionError):
        repository.create(_create_values(first_id, 1, "BUY"))
