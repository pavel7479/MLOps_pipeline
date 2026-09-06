import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.config.settings import (
    AppSettings, CatBoostSearchSpace, CatBoostSettings, EvaluationSettings,
    HttpSettings, HyperparameterSearchSettings, LightGBMSearchSpace,
    LightGBMSettings, LoggingSettings, MarketDataSettings, MLDatasetSettings,
    ModelsSettings, SearchSpacesSettings, StorageSettings,
    TimeSeriesValidationSettings, XGBoostSearchSpace, XGBoostSettings,
)
from src.models import LightGBMModel
from src.models.base import INVERSE_LABEL_MAPPING
from src.tuning import (
    HyperparameterTuner, HyperparameterTuningPipeline, ModelSearchResult,
    build_time_series_split, derive_gap_rows, describe_folds, sklearn_scoring_name,
)


def make_settings(tmp_path: Path, n_iter: int = 1) -> AppSettings:
    return AppSettings(
        market_data=MarketDataSettings(
            "binance", "BTCUSDT", "1h", datetime(2024, 1, 1, tzinfo=timezone.utc), None
        ),
        storage=StorageSettings(tmp_path / "raw", tmp_path / "processed", "parquet"),
        http=HttpSettings(1, 0, 0),
        logging=LoggingSettings("INFO"),
        ml_dataset=MLDatasetSettings(tmp_path / "ml"),
        models=ModelsSettings(
            catboost=CatBoostSettings(iterations=3, depth=2, learning_rate=0.1, random_seed=42),
            xgboost=XGBoostSettings(n_estimators=3, max_depth=2, learning_rate=0.1, random_state=42),
            lightgbm=LightGBMSettings(n_estimators=3, max_depth=2, learning_rate=0.1, random_state=42),
        ),
        evaluation=EvaluationSettings(
            "macro_f1", tmp_path / "models", tmp_path / "block3_evaluation"
        ),
        time_series_validation=TimeSeriesValidationSettings(n_splits=2),
        hyperparameter_search=HyperparameterSearchSettings(
            n_iter=n_iter,
            random_state=42,
            scoring="macro_f1",
            models_dir=tmp_path / "tuned_models",
            results_dir=tmp_path / "search_results",
        ),
        search_spaces=SearchSpacesSettings(
            catboost=CatBoostSearchSpace((3,), (2,), (0.1,), (1,)),
            xgboost=XGBoostSearchSpace((3,), (2,), (0.1,), (1.0,), (1.0,), (1,)),
            lightgbm=LightGBMSearchSpace((2, 3), (2,), (0.1,), (3,), (1.0,), (1.0,), (2,)),
        ),
    )


def test_time_series_split_is_chronological_and_honors_horizon_gap() -> None:
    assert derive_gap_rows(3, "1h") == 3
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"))
    splitter = build_time_series_split(n_splits=5, gap=3)
    descriptions = describe_folds(splitter, timestamps)
    assert getattr(splitter, "shuffle", False) is False
    assert len(descriptions) == 5
    for train_indices, validation_indices in splitter.split(timestamps):
        assert train_indices[-1] < validation_indices[0]
        assert validation_indices[0] - train_indices[-1] - 1 == 3
    assert all(fold["observed_gap_rows"] == 3 for fold in descriptions)


def test_search_uses_explicit_macro_f1_and_returns_best_parameters(tmp_path: Path) -> None:
    assert sklearn_scoring_name("macro_f1") == "f1_macro"
    settings = make_settings(tmp_path, n_iter=2)
    values, encoded = make_classification(
        n_samples=75, n_features=4, n_informative=3, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=2,
    )
    features = pd.DataFrame(values, columns=[f"f{i}" for i in range(4)])
    target = pd.Series([INVERSE_LABEL_MAPPING[int(value)] for value in encoded])
    tuner = HyperparameterTuner(settings)
    assert tuner._estimators_and_spaces()["lightgbm"][0].get_params()["subsample_freq"] == 1
    result = tuner.tune_model(
        "lightgbm", features, target, build_time_series_split(2, 3)
    )
    best_candidate = result.candidates.iloc[0]
    assert best_candidate["rank_test_score"] == 1
    assert result.mean_cv_macro_f1 == result.candidates["mean_test_score"].max()
    for parameter, value in result.best_params.items():
        assert best_candidate[f"param_{parameter}"] == value
    assert len(result.fold_scores) == 2
    assert np.isfinite(result.mean_cv_macro_f1)


class RecordingTuner:
    model_names = ("catboost", "xgboost", "lightgbm")

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def tune_model(self, model_name, features, target, splitter):
        self.events.append(f"tune_{model_name}")
        params = {
            "catboost": {"iterations": 3, "depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 1},
            "xgboost": {
                "n_estimators": 3, "max_depth": 2, "learning_rate": 0.1,
                "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
            },
            "lightgbm": {
                "n_estimators": 3, "max_depth": 2, "learning_rate": 0.1,
                "num_leaves": 3, "subsample": 1.0, "colsample_bytree": 1.0,
                "min_child_samples": 2,
            },
        }[model_name]
        candidates = pd.DataFrame([{
            "rank_test_score": 1, "mean_test_score": 0.3, "std_test_score": 0.01,
            "mean_fit_time": 0.01, "mean_score_time": 0.001, "params": params,
            "split0_test_score": 0.29, "split1_test_score": 0.31,
        }])
        return ModelSearchResult(
            model_name, params, 0.3, 0.01, [0.29, 0.31], 0.02, candidates
        )


def test_pipeline_defers_validation_never_reads_test_and_saves_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    settings = make_settings(tmp_path)
    values, encoded = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=4,
    )
    feature_names = [f"feature_{index}" for index in range(4)]
    frame = pd.DataFrame(values, columns=feature_names)
    frame["target"] = [INVERSE_LABEL_MAPPING[int(value)] for value in encoded]
    train = frame.iloc[:90].copy()
    validation = frame.iloc[90:].copy()
    train.insert(0, "timestamp", pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC"))
    validation.insert(
        0, "timestamp", pd.date_range("2024-01-06", periods=30, freq="h", tz="UTC")
    )
    settings.ml_dataset.output_dir.mkdir(parents=True)
    train.to_parquet(settings.ml_dataset.output_dir / "BTCUSDT_1h_train.parquet", index=False)
    validation.to_parquet(
        settings.ml_dataset.output_dir / "BTCUSDT_1h_validation.parquet", index=False
    )
    validation.to_parquet(
        settings.ml_dataset.output_dir / "BTCUSDT_1h_test.parquet", index=False
    )
    (settings.ml_dataset.output_dir / "feature_manifest.json").write_text(
        json.dumps({"features": feature_names, "target": "target"}), encoding="utf-8"
    )
    settings.evaluation.results_dir.mkdir(parents=True)
    (settings.evaluation.results_dir / "model_comparison.json").write_text(
        json.dumps({"models": [
            {"model": "catboost", "macro_f1": 0.39},
            {"model": "xgboost", "macro_f1": 0.38},
            {"model": "lightgbm", "macro_f1": 0.40},
        ]}),
        encoding="utf-8",
    )

    events: list[str] = []
    original_read_parquet = pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        name = Path(path).name
        assert "_test.parquet" not in name
        event = "read_validation" if "_validation.parquet" in name else "read_train"
        events.append(event)
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    report = HyperparameterTuningPipeline(
        settings, tuner=RecordingTuner(events)
    ).run()

    assert events == [
        "read_train", "tune_catboost", "tune_xgboost", "tune_lightgbm",
        "read_validation",
    ]
    assert report.gap_rows == 3 and report.n_splits == 2
    assert len(report.validation_results) == 3
    assert all(path.is_file() for path in report.artifact_paths.values())
    metadata = json.loads(report.artifact_paths["tuning_metadata"].read_text(encoding="utf-8"))
    assert metadata["test_dataset_used"] is False
    assert metadata["validation_used_during_search"] is False
    assert metadata["sklearn_scoring"] == "f1_macro"
    comparison = pd.read_csv(report.artifact_paths["validation_comparison_csv"])
    assert set(comparison["model"]) == set(RecordingTuner.model_names)
    assert "tuned_validation_accuracy" in comparison
    assert report.artifact_paths["best_params"].name == "best_params.json"
    assert report.artifact_paths["cv_summary"].name == "cv_summary.json"
    loaded = LightGBMModel.load(report.artifact_paths["model_lightgbm"])
    assert len(loaded.predict(validation[feature_names])) == len(validation)
