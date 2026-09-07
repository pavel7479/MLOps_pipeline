from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.config import MLflowSettings, load_settings
from src.mlops import MLflowTracker, MLflowTrackingError, ModelRegistry, sha256_file
from src.models import DummyModel
from src.config.settings import DummySettings


def _local_settings(tmp_path: Path) -> MLflowSettings:
    database = (tmp_path / "mlflow.db").resolve().as_posix()
    artifacts = (tmp_path / "artifacts").resolve()
    artifacts.mkdir()
    return MLflowSettings(
        tracking_uri=f"sqlite:///{database}",
        experiment_name="test_crypto_experiment",
        registered_model_name="test_crypto_classifier",
        artifact_location=artifacts.as_uri(),
        enabled=True,
    )


def _fitted_dummy(strategy: str = "most_frequent"):
    features = pd.DataFrame(
        {"return_1": [0.0, 0.1, -0.1, 0.2, -0.2, 0.0], "rsi": [50, 55, 45, 60, 40, 51]}
    )
    target = pd.Series(["HOLD", "BUY", "SELL", "BUY", "SELL", "HOLD"])
    model = DummyModel(DummySettings(strategy=strategy)).fit(features, target)
    return model, features


def test_project_config_enables_named_mlflow() -> None:
    settings = load_settings(Path(__file__).resolve().parents[1] / "config.yaml")
    assert settings.mlflow.enabled is True
    assert settings.mlflow.tracking_uri == "http://127.0.0.1:5000"
    assert settings.mlflow.experiment_name == "crypto_direction_classification"
    assert settings.mlflow.registered_model_name == "crypto_direction_classifier"


def test_disabled_tracker_is_a_noop(monkeypatch) -> None:
    tracker = MLflowTracker(MLflowSettings(enabled=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled tracker called MLflow")

    monkeypatch.setattr(mlflow, "start_run", forbidden)
    monkeypatch.setattr(mlflow, "log_params", forbidden)
    tracker.check_health()
    assert tracker.start_run("disabled") is None
    tracker.log_params({"model_name": "dummy"})
    tracker.log_metrics({"validation_macro_f1": 0.1})
    tracker.log_tags({"stage": "test"})
    tracker.end_run()


def test_tracker_forwards_parameters_metrics_and_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    tracker = MLflowTracker(MLflowSettings(enabled=True))
    calls: dict[str, object] = {}
    artifact = tmp_path / "feature_manifest.json"
    artifact.write_text('{"features": ["x"]}', encoding="utf-8")
    monkeypatch.setattr(mlflow, "log_params", lambda values: calls.setdefault("params", values))
    monkeypatch.setattr(mlflow, "log_metrics", lambda values: calls.setdefault("metrics", values))
    monkeypatch.setattr(
        mlflow,
        "log_artifact",
        lambda path, artifact_path=None: calls.setdefault(
            "artifact", (path, artifact_path)
        ),
    )
    monkeypatch.setattr(
        mlflow,
        "log_dict",
        lambda payload, artifact_file: calls.setdefault(
            "metadata", (payload, artifact_file)
        ),
    )
    tracker.log_params({"model_name": "lightgbm", "n_estimators": 300})
    tracker.log_metrics(
        {
            "validation_accuracy": 0.4,
            "validation_macro_f1": 0.39,
            "buy_f1": 0.3,
            "hold_f1": 0.5,
            "sell_recall": 0.2,
        }
    )
    tracker.log_artifact(artifact, "dataset")
    tracker.log_dict({"features_count": 1}, "dataset/dataset_metadata.json")
    assert calls["params"] == {"model_name": "lightgbm", "n_estimators": 300}
    assert calls["metrics"] == {
        "validation_accuracy": 0.4,
        "validation_macro_f1": 0.39,
        "buy_f1": 0.3,
        "hold_f1": 0.5,
        "sell_recall": 0.2,
    }
    assert calls["artifact"] == (str(artifact), "dataset")
    assert calls["metadata"] == (
        {"features_count": 1},
        "dataset/dataset_metadata.json",
    )


def test_sha256_is_stable_and_changes_with_content(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"same dataset")
    second.write_bytes(b"same dataset")
    assert sha256_file(first) == sha256_file(second)
    second.write_bytes(b"changed dataset")
    assert sha256_file(first) != sha256_file(second)


def test_unavailable_server_error_is_clear(monkeypatch) -> None:
    tracker = MLflowTracker(
        MLflowSettings(tracking_uri="http://127.0.0.1:9", enabled=True)
    )

    import httpx

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            httpx.ConnectError("connection refused")
        ),
    )
    with pytest.raises(
        MLflowTrackingError,
        match=r"MLflow tracking server is unavailable: http://127\.0\.0\.1:9",
    ):
        tracker.check_health(timeout_seconds=0.01)


def test_registry_versions_alias_reassignment_and_load_by_alias(tmp_path: Path) -> None:
    mlflow.end_run()
    settings = _local_settings(tmp_path)
    tracker = MLflowTracker(settings)
    tracker.check_health()
    model, features = _fitted_dummy()
    feature_names = list(features)

    first_run = tracker.start_run("first")
    tracker.log_params({"model_name": "dummy", "strategy": "most_frequent"})
    tracker.log_metrics(
        {"validation_accuracy": 0.34, "validation_macro_f1": 0.17, "sell_recall": 0.0}
    )
    _, first_loaded_predictions = tracker.log_model(
        model, feature_names, features.head(3)
    )
    tracker.end_run()
    assert np.array_equal(first_loaded_predictions, model.predict(features.head(3)))

    second_run = tracker.start_run("second")
    tracker.log_model(model, feature_names, features.tail(3))
    tracker.end_run()
    assert first_run is not None and second_run is not None

    registry = ModelRegistry(settings)
    first_version = registry.register_run_model(
        first_run,
        description="first test version",
        tags={"source_stage": "test_1"},
    )
    second_version = registry.register_run_model(
        second_run,
        description="second test version",
        tags={"source_stage": "test_2"},
    )
    registry.set_champion(first_version.version)
    registry.set_challenger(second_version.version)
    assert registry.get_alias("champion").version == first_version.version
    assert registry.get_alias("challenger").version == second_version.version

    registry.set_champion(second_version.version)
    assert registry.get_alias("champion").version == second_version.version
    loaded = registry.load_champion_model()
    assert np.array_equal(
        np.asarray(loaded.predict(features.head(3))),
        model.predict(features.head(3)),
    )
    registered = registry.client.get_registered_model(settings.registered_model_name)
    assert registered.name == settings.registered_model_name
    versions = registry.client.search_model_versions(
        f"name='{settings.registered_model_name}'"
    )
    assert {version.version for version in versions} == {
        first_version.version,
        second_version.version,
    }
