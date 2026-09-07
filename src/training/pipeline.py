"""Orchestration for fair Train/Validation model comparison."""

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import AppSettings
from src.dataset.storage import MLDatasetStorage
from src.models import (
    BaseClassifier,
    CatBoostModel,
    DummyModel,
    LABEL_MAPPING,
    LABEL_ORDER,
    LightGBMModel,
    LogisticRegressionModel,
    XGBoostModel,
)
from src.mlops import MLflowTracker
from src.mlops.integration import log_baseline_model

from .evaluator import evaluate_predictions, select_best_model
from .models import ModelTrainingReport
from .validator import TrainingDataValidator

LOGGER = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Train five classifiers on Train and select only on Validation."""

    def __init__(
        self, settings: AppSettings, tracker: MLflowTracker | None = None
    ) -> None:
        self.settings = settings
        self.validator = TrainingDataValidator()
        self.storage = MLDatasetStorage()
        self.tracker = tracker or MLflowTracker(settings.mlflow)

    @property
    def dataset_paths(self) -> dict[str, Path]:
        stem = f"{self.settings.market_data.symbol}_{self.settings.market_data.timeframe}"
        root = self.settings.ml_dataset.output_dir
        return {
            "train": root / f"{stem}_train.parquet",
            "validation": root / f"{stem}_validation.parquet",
            "test": root / f"{stem}_test.parquet",
            "manifest": root / "feature_manifest.json",
        }

    def _load_inputs(self) -> tuple[pd.DataFrame, pd.DataFrame, list[str], bool]:
        paths = self.dataset_paths
        for key in ("train", "validation", "manifest"):
            if not paths[key].is_file():
                raise FileNotFoundError(f"Required {key} artifact not found: {paths[key]}")
        test_exists = paths["test"].is_file()
        if not test_exists:
            LOGGER.warning("Reserved Test file is absent; it is not required or read by Block 3")
        train = pd.read_parquet(paths["train"])
        validation = pd.read_parquet(paths["validation"])
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        feature_names = manifest.get("features")
        if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
            raise ValueError("feature_manifest.json must contain a string list named features")
        self.validator.validate(train, validation, feature_names)
        return train, validation, feature_names, test_exists

    def _build_models(self) -> list[BaseClassifier]:
        models = self.settings.models
        return [
            DummyModel(models.dummy),
            LogisticRegressionModel(models.logistic_regression),
            CatBoostModel(models.catboost),
            XGBoostModel(models.xgboost),
            LightGBMModel(models.lightgbm),
        ]

    def _save_evaluation_artifacts(
        self,
        results: list[dict[str, Any]],
        feature_importance: dict[str, list[dict[str, float | str]]],
        coefficients: dict[str, list[dict[str, float | str]]],
        best: dict[str, Any],
        train_rows: int,
        validation_rows: int,
        feature_names: list[str],
        test_exists: bool,
    ) -> dict[str, Path]:
        output = self.settings.evaluation.results_dir
        output.mkdir(parents=True, exist_ok=True)
        comparison_keys = [
            "model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "training_time_seconds", "improvement_over_dummy",
        ]
        comparison = [{key: result[key] for key in comparison_keys} for result in results]
        comparison_path = output / "model_comparison.csv"
        temporary_csv = comparison_path.with_suffix(".csv.tmp")
        pd.DataFrame(comparison).to_csv(temporary_csv, index=False)
        temporary_csv.replace(comparison_path)
        paths = {
            "comparison_csv": comparison_path,
            "comparison_json": self.storage.save_json({"models": comparison}, output / "model_comparison.json"),
            "class_metrics": self.storage.save_json(
                {result["model"]: result["class_metrics"] for result in results}, output / "class_metrics.json"
            ),
            "confusion_matrices": self.storage.save_json(
                {result["model"]: result["confusion_matrix"] for result in results},
                output / "confusion_matrices.json",
            ),
            "feature_importance": self.storage.save_json(feature_importance, output / "feature_importance.json"),
            "logistic_coefficients": self.storage.save_json(
                coefficients, output / "logistic_coefficients.json"
            ),
            "label_mapping": self.storage.save_json(
                {"mapping": LABEL_MAPPING, "probability_column_order": list(LABEL_ORDER)},
                output / "label_mapping.json",
            ),
            "best_model": self.storage.save_json(
                {
                    "model": best["model"], "selection_dataset": "validation",
                    "selection_metric": self.settings.evaluation.primary_metric,
                    "score": best[self.settings.evaluation.primary_metric],
                },
                output / "best_model.json",
            ),
        }
        model_parameters = {model.name: model.parameters for model in self._build_models()}
        metadata = {
            "train_dataset_path": str(self.dataset_paths["train"]),
            "validation_dataset_path": str(self.dataset_paths["validation"]),
            "test_dataset_path": str(self.dataset_paths["test"]),
            "test_dataset_exists": test_exists,
            "test_dataset_used": False,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "features_count": len(feature_names),
            "features": feature_names,
            "label_mapping": LABEL_MAPPING,
            "model_parameters": model_parameters,
            "random_seeds": {
                "logistic_regression": self.settings.models.logistic_regression.random_state,
                "catboost": self.settings.models.catboost.random_seed,
                "xgboost": self.settings.models.xgboost.random_state,
                "lightgbm": self.settings.models.lightgbm.random_state,
            },
        }
        paths["training_metadata"] = self.storage.save_json(metadata, output / "training_metadata.json")
        return paths

    def run(self) -> ModelTrainingReport:
        """Train, evaluate, persist, compare, and return validation results."""
        LOGGER.info("Starting model training pipeline; Test is reserved and will not be read")
        self.tracker.check_health()
        train, validation, feature_names, test_exists = self._load_inputs()
        LOGGER.info(
            "Training inputs: train=%d validation=%d features=%d",
            len(train), len(validation), len(feature_names),
        )
        train_features = train[feature_names]
        validation_features = validation[feature_names]
        train_target = train["target"]
        validation_target = validation["target"].to_numpy()
        results: list[dict[str, Any]] = []
        importances: dict[str, list[dict[str, float | str]]] = {}
        coefficients: dict[str, list[dict[str, float | str]]] = {}
        model_paths: dict[str, Path] = {}

        for model in self._build_models():
            LOGGER.info("Training %s", model.name)
            started = perf_counter()
            model.fit(train_features, train_target)
            elapsed = perf_counter() - started
            predicted = model.predict(validation_features)
            probabilities = model.predict_proba(validation_features)
            if probabilities.shape != (len(validation), len(LABEL_ORDER)) or not np.isfinite(probabilities).all():
                raise ValueError(f"{model.name} returned invalid class probabilities")
            evaluation = evaluate_predictions(model.name, validation_target, predicted)
            evaluation["training_time_seconds"] = elapsed
            results.append(evaluation)
            importances[model.name] = model.feature_importance(feature_names)
            model_coefficients = model.coefficient_report(feature_names)
            if model_coefficients:
                coefficients = model_coefficients
            model_path = self.settings.evaluation.models_dir / f"{model.name}.joblib"
            model.save(model_path)
            loaded = type(model).load(model_path)
            if not np.array_equal(predicted, loaded.predict(validation_features)):
                raise RuntimeError(f"{model.name} predictions changed after save/load")
            model_paths[model.name] = model_path
            log_baseline_model(
                self.tracker,
                self.settings,
                model,
                evaluation,
                train,
                validation,
                feature_names,
                model_path,
                importances[model.name],
                model_coefficients,
            )
            LOGGER.info(
                "%s trained in %.3fs; validation macro_f1=%.6f; saved=%s",
                model.name, elapsed, evaluation["macro_f1"], model_path,
            )

        dummy_score = next(float(result["macro_f1"]) for result in results if result["model"] == "dummy")
        for result in results:
            result["improvement_over_dummy"] = float(result["macro_f1"]) - dummy_score
        results.sort(key=lambda result: float(result[self.settings.evaluation.primary_metric]), reverse=True)
        best = select_best_model(results, self.settings.evaluation.primary_metric)
        artifact_paths = self._save_evaluation_artifacts(
            results, importances, coefficients, best, len(train), len(validation), feature_names, test_exists
        )
        artifact_paths.update({f"model_{name}": path for name, path in model_paths.items()})
        LOGGER.info(
            "Training pipeline complete; best validation model=%s macro_f1=%.6f",
            best["model"], best["macro_f1"],
        )
        return ModelTrainingReport(
            train_rows=len(train), validation_rows=len(validation), features_count=len(feature_names),
            results=results, best_model=str(best["model"]), best_score=float(best["macro_f1"]),
            dummy_score=dummy_score, artifact_paths=artifact_paths,
        )
