"""Project-specific translation from existing artifacts to MLflow runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

from src.config.settings import AppSettings
from src.models import BaseClassifier, LABEL_MAPPING, LABEL_ORDER

from .mlflow_tracker import MLflowTracker
from .models import dataframe_period, git_metadata, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FAMILIES = {
    "dummy": "sklearn",
    "logistic_regression": "sklearn",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}


def _json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required MLflow import artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_paths(settings: AppSettings) -> dict[str, Path]:
    stem = f"{settings.market_data.symbol}_{settings.market_data.timeframe}"
    root = settings.ml_dataset.output_dir
    return {
        "train": root / f"{stem}_train.parquet",
        "validation": root / f"{stem}_validation.parquet",
        "manifest": root / "feature_manifest.json",
    }


def _load_dataset_context(
    settings: AppSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Path]]:
    paths = _dataset_paths(settings)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required MLflow dataset artifact not found: {path}")
    manifest = _json(paths["manifest"])
    feature_names = manifest.get("features")
    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise ValueError("feature_manifest.json must contain a string list named features")
    train = pd.read_parquet(paths["train"])
    validation = pd.read_parquet(paths["validation"])
    return train, validation, feature_names, paths


def _tags(settings: AppSettings, model_name: str, origin: str) -> dict[str, str]:
    return {
        "project": "crypto_ml_platform",
        "task": "multiclass_classification",
        "symbol": settings.market_data.symbol,
        "timeframe": settings.market_data.timeframe,
        "dataset_stage": "validation",
        "pipeline_stage": "block_6",
        "model_family": MODEL_FAMILIES[model_name],
        "run_origin": origin,
        "mlflow_version": mlflow.__version__,
        **git_metadata(PROJECT_ROOT),
    }


def _fingerprints(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "train_dataset_sha256": sha256_file(paths["train"]),
        "validation_dataset_sha256": sha256_file(paths["validation"]),
        "feature_manifest_sha256": sha256_file(paths["manifest"]),
    }


def _common_params(
    settings: AppSettings,
    model_name: str,
    model_parameters: dict[str, Any],
    train_rows: int,
    validation_rows: int,
    features_count: int,
) -> dict[str, Any]:
    random_seed = model_parameters.get(
        "random_seed",
        model_parameters.get("random_state", settings.hyperparameter_search.random_state),
    )
    return {
        "model_name": model_name,
        "random_seed": random_seed,
        "features_count": features_count,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "target_horizon_hours": settings.target.horizon_hours,
        "buy_threshold": settings.target.buy_threshold,
        "sell_threshold": settings.target.sell_threshold,
        **model_parameters,
    }


def _validation_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "validation_accuracy": float(evaluation["accuracy"]),
        "validation_macro_precision": float(evaluation["macro_precision"]),
        "validation_macro_recall": float(evaluation["macro_recall"]),
        "validation_macro_f1": float(evaluation["macro_f1"]),
        "training_time_seconds": float(evaluation["training_time_seconds"]),
    }
    for label in LABEL_ORDER:
        values = evaluation["class_metrics"][label]
        prefix = label.lower()
        metrics[f"{prefix}_precision"] = float(values["precision"])
        metrics[f"{prefix}_recall"] = float(values["recall"])
        metrics[f"{prefix}_f1"] = float(values["f1"])
    return metrics


def _backtest_row(settings: AppSettings, strategy_slug: str) -> dict[str, Any] | None:
    path = settings.backtesting.output_dir / "backtest_summary.json"
    if not path.is_file():
        return None
    strategies = _json(path).get("strategies", [])
    return next(
        (row for row in strategies if row.get("strategy_slug") == strategy_slug),
        None,
    )


def _backtest_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        "backtest_return_pct": float(row["total_return_pct"]),
        "backtest_max_drawdown_pct": float(row["max_drawdown_pct"]),
        "backtest_sharpe": float(row["sharpe_ratio"]),
        "backtest_trades": float(row["number_of_trades"]),
        "backtest_win_rate": float(row["win_rate_pct"]),
        "backtest_total_fees": float(row["total_fees"]),
        "backtest_buy_hold_return_pct": float(row["buy_and_hold_return_pct"]),
    }


def _log_backtest_artifacts(
    tracker: MLflowTracker, settings: AppSettings, strategy_slug: str
) -> None:
    for name in (
        f"equity_curve_{strategy_slug}.csv",
        f"trades_{strategy_slug}.csv",
        "backtest_metadata.json",
    ):
        path = settings.backtesting.output_dir / name
        if path.is_file():
            tracker.log_artifact(path, "backtesting")


def _dataset_metadata(
    settings: AppSettings,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
    paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "train": {**dataframe_period(train), "rows": len(train), "path": str(paths["train"])},
        "validation": {
            **dataframe_period(validation),
            "rows": len(validation),
            "path": str(paths["validation"]),
        },
        "features_count": len(feature_names),
        "features": feature_names,
        "target": {
            "horizon_hours": settings.target.horizon_hours,
            "buy_threshold": settings.target.buy_threshold,
            "sell_threshold": settings.target.sell_threshold,
            "prediction_timestamp_semantics": (
                "features at t use the fully closed candle t; predict after its close; "
                "target is Close[t+horizon] relative to Close[t]"
            ),
        },
        "fingerprints": _fingerprints(paths),
        "test_dataset_used": False,
        "mlflow_version": mlflow.__version__,
    }


def log_baseline_model(
    tracker: MLflowTracker,
    settings: AppSettings,
    model: BaseClassifier,
    evaluation: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
    model_path: Path,
    feature_importance: list[dict[str, Any]],
    coefficients: dict[str, Any],
) -> str | None:
    """Create one complete baseline run after successful local persistence."""
    if not tracker.enabled:
        return None
    run_id = tracker.start_run(
        f"baseline_{model.name}", _tags(settings, model.name, "direct_training")
    )
    try:
        paths = _dataset_paths(settings)
        tracker.log_params(
            {
                **_common_params(
                    settings,
                    model.name,
                    model.parameters,
                    len(train),
                    len(validation),
                    len(feature_names),
                ),
                **_fingerprints(paths),
            }
        )
        tracker.log_metrics(_validation_metrics(evaluation))
        tracker.log_artifact(paths["manifest"], "dataset")
        tracker.log_dict(
            _dataset_metadata(settings, train, validation, feature_names, paths),
            "dataset/dataset_metadata.json",
        )
        tracker.log_dict(
            {"mapping": LABEL_MAPPING, "probability_column_order": list(LABEL_ORDER)},
            "evaluation/label_mapping.json",
        )
        tracker.log_dict(evaluation["class_metrics"], "evaluation/class_metrics.json")
        tracker.log_dict(evaluation["confusion_matrix"], "evaluation/confusion_matrix.json")
        tracker.log_dict(feature_importance, "evaluation/feature_importance.json")
        if coefficients:
            tracker.log_dict(coefficients, "evaluation/logistic_coefficients.json")
        tracker.log_artifact(model_path, "native_model")
        if model.name == "lightgbm":
            backtest = _backtest_row(settings, "lightgbm_block3")
            if backtest is not None:
                tracker.log_metrics(_backtest_metrics(backtest))
                _log_backtest_artifacts(tracker, settings, "lightgbm_block3")
        tracker.log_model(
            model,
            feature_names,
            validation[feature_names].head(5),
            code_paths=[str(PROJECT_ROOT / "src")],
        )
    except Exception:
        tracker.end_run("FAILED")
        raise
    tracker.end_run()
    return run_id


def log_tuning_runs_from_artifacts(
    settings: AppSettings,
    tracker: MLflowTracker,
    *,
    origin: str,
) -> dict[str, str]:
    """Log saved Block 4 results without repeating RandomizedSearchCV."""
    if not tracker.enabled:
        return {}
    train, validation, feature_names, dataset_paths = _load_dataset_context(settings)
    root = settings.hyperparameter_search.results_dir
    comparison = {
        row["model"]: row for row in _json(root / "validation_comparison.json")["models"]
    }
    cv_summary = _json(root / "cv_summary.json")
    cv_models = cv_summary["models"]
    class_metrics = _json(root / "validation_class_metrics.json")
    confusion_matrices = _json(root / "validation_confusion_matrices.json")
    run_ids: dict[str, str] = {}

    for model_name in ("catboost", "xgboost", "lightgbm"):
        model_path = (
            settings.hyperparameter_search.models_dir / f"{model_name}_tuned.joblib"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"Tuned model not found: {model_path}")
        model = joblib.load(model_path)
        if not isinstance(model, BaseClassifier):
            raise TypeError(f"Unexpected tuned model type: {type(model).__name__}")
        row = comparison[model_name]
        cv = cv_models[model_name]
        evaluation = {
            "accuracy": row["tuned_validation_accuracy"],
            "macro_precision": row["tuned_validation_macro_precision"],
            "macro_recall": row["tuned_validation_macro_recall"],
            "macro_f1": row["tuned_validation_macro_f1"],
            "training_time_seconds": row["final_training_time_seconds"],
            "class_metrics": class_metrics[model_name],
            "confusion_matrix": confusion_matrices[model_name],
        }
        run_id = tracker.start_run(
            f"tuned_{model_name}", _tags(settings, model_name, origin)
        )
        try:
            tracker.log_params(
                {
                    **_common_params(
                        settings,
                        model_name,
                        model.parameters,
                        len(train),
                        len(validation),
                        len(feature_names),
                    ),
                    **_fingerprints(dataset_paths),
                    "n_splits": settings.time_series_validation.n_splits,
                    "gap": cv_summary["gap_rows"],
                    "search_n_iter": settings.hyperparameter_search.n_iter,
                    "scoring": "f1_macro",
                }
            )
            tracker.log_metrics(
                {
                    **_validation_metrics(evaluation),
                    "cv_mean_macro_f1": float(cv["mean_cv_macro_f1"]),
                    "cv_std_macro_f1": float(cv["std_cv_macro_f1"]),
                    "search_time_seconds": float(cv["search_duration_seconds"]),
                }
            )
            tracker.log_artifact(dataset_paths["manifest"], "dataset")
            tracker.log_dict(
                _dataset_metadata(
                    settings, train, validation, feature_names, dataset_paths
                ),
                "dataset/dataset_metadata.json",
            )
            tracker.log_dict(
                class_metrics[model_name], "evaluation/class_metrics.json"
            )
            tracker.log_dict(
                confusion_matrices[model_name], "evaluation/confusion_matrix.json"
            )
            tracker.log_dict(row, "evaluation/validation_comparison.json")
            for suffix in ("best_params.json", "cv_summary.json", "search_results.csv"):
                tracker.log_artifact(root / f"{model_name}_{suffix}", "tuning")
            tracker.log_artifact(root / "tuning_metadata.json", "tuning")
            tracker.log_artifact(model_path, "native_model")
            if model_name == "lightgbm":
                backtest = _backtest_row(settings, "lightgbm_tuned")
                if backtest is not None:
                    tracker.log_metrics(_backtest_metrics(backtest))
                    _log_backtest_artifacts(tracker, settings, "lightgbm_tuned")
            tracker.log_model(
                model,
                feature_names,
                validation[feature_names].head(5),
                code_paths=[str(PROJECT_ROOT / "src")],
            )
        except Exception:
            tracker.end_run("FAILED")
            raise
        tracker.end_run()
        if run_id is not None:
            run_ids[model_name] = run_id
    return run_ids


def log_buy_and_hold_run(settings: AppSettings, tracker: MLflowTracker) -> str | None:
    """Keep the market benchmark separate from classifier quality metrics."""
    row = _backtest_row(settings, "buy_and_hold")
    if not tracker.enabled or row is None:
        return None
    run_id = tracker.start_run(
        "buy_and_hold_backtest",
        {
            "project": "crypto_ml_platform",
            "task": "backtesting_baseline",
            "symbol": settings.market_data.symbol,
            "timeframe": settings.market_data.timeframe,
            "dataset_stage": "validation",
            "pipeline_stage": "block_6",
            "run_origin": "imported_existing_experiment",
            "mlflow_version": mlflow.__version__,
            **git_metadata(PROJECT_ROOT),
        },
    )
    try:
        tracker.log_params(
            {
                "strategy": "buy_and_hold",
                "initial_cash": row["initial_cash"],
                "commission_rate": settings.backtesting.commission_rate,
            }
        )
        tracker.log_metrics(
            {
                "backtest_return_pct": float(row["total_return_pct"]),
                "backtest_max_drawdown_pct": float(row["max_drawdown_pct"]),
                "backtest_sharpe": float(row["sharpe_ratio"]),
                "backtest_trades": float(row["number_of_trades"]),
                "backtest_total_fees": float(row["total_fees"]),
            }
        )
        _log_backtest_artifacts(tracker, settings, "buy_and_hold")
    except Exception:
        tracker.end_run("FAILED")
        raise
    tracker.end_run()
    return run_id
