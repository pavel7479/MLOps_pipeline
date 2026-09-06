import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from src.config.settings import (
    AppSettings, CatBoostSettings, DummySettings, EvaluationSettings, HttpSettings,
    LightGBMSettings, LoggingSettings, LogisticRegressionSettings, MarketDataSettings,
    MLDatasetSettings, ModelsSettings, StorageSettings, XGBoostSettings,
)
from src.models.base import INVERSE_LABEL_MAPPING
from src.training.pipeline import ModelTrainingPipeline


def test_full_training_pipeline_never_reads_test(tmp_path: Path, monkeypatch) -> None:
    values, encoded = make_classification(
        n_samples=150, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=7,
    )
    feature_names = [f"feature_{index}" for index in range(5)]
    data = pd.DataFrame(values, columns=feature_names)
    data["target"] = [INVERSE_LABEL_MAPPING[int(value)] for value in encoded]
    train = data.iloc[:100].copy()
    validation = data.iloc[100:].copy()
    train.insert(0, "timestamp", pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"))
    validation.insert(0, "timestamp", pd.date_range("2024-01-06", periods=50, freq="h", tz="UTC"))
    ml_dir = tmp_path / "ml"
    ml_dir.mkdir()
    train.to_parquet(ml_dir / "BTCUSDT_1h_train.parquet", index=False)
    validation.to_parquet(ml_dir / "BTCUSDT_1h_validation.parquet", index=False)
    (ml_dir / "feature_manifest.json").write_text(
        json.dumps({"features": feature_names, "target": "target"}), encoding="utf-8"
    )
    original_read_parquet = pd.read_parquet
    read_names: list[str] = []

    def guarded_read_parquet(path, *args, **kwargs):
        name = Path(path).name
        assert "test" not in name
        read_names.append(name)
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read_parquet)
    settings = AppSettings(
        market_data=MarketDataSettings(
            "binance", "BTCUSDT", "1h", datetime(2024, 1, 1, tzinfo=timezone.utc), None
        ),
        storage=StorageSettings(tmp_path / "raw", tmp_path / "processed", "parquet"),
        http=HttpSettings(1, 0, 0),
        logging=LoggingSettings("INFO"),
        ml_dataset=MLDatasetSettings(ml_dir),
        models=ModelsSettings(
            dummy=DummySettings(),
            logistic_regression=LogisticRegressionSettings(max_iter=200, random_state=42),
            catboost=CatBoostSettings(iterations=4, depth=2, learning_rate=0.1, random_seed=42),
            xgboost=XGBoostSettings(n_estimators=4, max_depth=2, learning_rate=0.1, random_state=42),
            lightgbm=LightGBMSettings(n_estimators=4, max_depth=2, learning_rate=0.1, random_state=42),
        ),
        evaluation=EvaluationSettings("macro_f1", tmp_path / "models", tmp_path / "evaluation"),
    )
    report = ModelTrainingPipeline(settings).run()
    assert read_names == ["BTCUSDT_1h_train.parquet", "BTCUSDT_1h_validation.parquet"]
    assert len(report.results) == 5
    assert report.best_model in {result["model"] for result in report.results}
    assert all((tmp_path / "models" / f"{name}.joblib").is_file() for name in (
        "dummy", "logistic_regression", "catboost", "xgboost", "lightgbm"
    ))
    assert all(path.is_file() for path in report.artifact_paths.values())
    metadata = json.loads(report.artifact_paths["training_metadata"].read_text(encoding="utf-8"))
    assert metadata["test_dataset_used"] is False
    assert metadata["train_rows"] == 100 and metadata["validation_rows"] == 50
    assert metadata["features"] == feature_names
