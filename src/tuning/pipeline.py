"""Block 4 orchestration: Train-only tuning and Validation comparison."""

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import AppSettings
from src.dataset.storage import MLDatasetStorage
from src.models import BaseClassifier, CatBoostModel, LABEL_ORDER, LightGBMModel, XGBoostModel
from src.training.evaluator import evaluate_predictions, select_best_model
from src.training.validator import TrainingDataValidator

from .models import ModelSearchResult, TuningReport
from .time_series import build_time_series_split, derive_gap_rows, describe_folds
from .tuner import HyperparameterTuner, sklearn_scoring_name

LOGGER = logging.getLogger(__name__)


class HyperparameterTuningPipeline:
    """Tune on Train folds, then evaluate fitted winners once on Validation."""

    def __init__(self, settings: AppSettings, tuner: HyperparameterTuner | None = None) -> None:
        self.settings = settings
        self.tuner = tuner or HyperparameterTuner(settings)
        self.validator = TrainingDataValidator()
        self.storage = MLDatasetStorage()

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

    def _load_train(self) -> tuple[pd.DataFrame, list[str], bool]:
        paths = self.dataset_paths
        for key in ("train", "manifest"):
            if not paths[key].is_file():
                raise FileNotFoundError(f"Required {key} artifact not found: {paths[key]}")
        baseline = self.settings.evaluation.results_dir / "model_comparison.json"
        if not baseline.is_file():
            raise FileNotFoundError(f"Block 3 comparison artifact not found: {baseline}")
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        feature_names = manifest.get("features")
        if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
            raise ValueError("feature_manifest.json must contain a string list named features")
        train = pd.read_parquet(paths["train"])
        self.validator.validate_frame(train, feature_names, "train")
        return train, feature_names, paths["test"].is_file()

    def _load_validation(
        self, train: pd.DataFrame, feature_names: list[str]
    ) -> pd.DataFrame:
        path = self.dataset_paths["validation"]
        if not path.is_file():
            raise FileNotFoundError(f"Required validation artifact not found: {path}")
        validation = pd.read_parquet(path)
        self.validator.validate_frame(validation, feature_names, "validation")
        self.validator.validate(train, validation, feature_names)
        return validation

    def _load_block3_scores(self) -> dict[str, float]:
        path = self.settings.evaluation.results_dir / "model_comparison.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = payload.get("models")
        if not isinstance(models, list):
            raise ValueError("Block 3 model_comparison.json must contain a models list")
        scores = {
            str(row["model"]): float(row["macro_f1"])
            for row in models
            if row.get("model") in self.tuner.model_names
        }
        missing = set(self.tuner.model_names) - set(scores)
        if missing:
            raise ValueError(f"Block 3 comparison lacks models: {sorted(missing)}")
        return scores

    def _build_tuned_model(self, name: str, parameters: dict[str, Any]) -> BaseClassifier:
        builders = {
            "catboost": lambda: CatBoostModel(self.settings.models.catboost),
            "xgboost": lambda: XGBoostModel(self.settings.models.xgboost),
            "lightgbm": lambda: LightGBMModel(self.settings.models.lightgbm),
        }
        try:
            model = builders[name]()
        except KeyError as exc:
            raise ValueError(f"Unsupported tuned model: {name}") from exc
        model.estimator.set_params(**parameters)
        model.parameters.update(parameters)
        if name == "lightgbm":
            model.estimator.set_params(subsample_freq=1)
            model.parameters["subsample_freq"] = 1
        return model

    @staticmethod
    def _save_csv(frame: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        return path

    def _save_artifacts(
        self,
        searches: list[ModelSearchResult],
        validation_results: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        best: dict[str, Any],
        fold_details: list[dict[str, Any]],
        train_rows: int,
        validation_rows: int,
        feature_names: list[str],
        gap_rows: int,
        test_exists: bool,
        model_paths: dict[str, Path],
    ) -> dict[str, Path]:
        output = self.settings.hyperparameter_search.results_dir
        output.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for search in searches:
            paths[f"{search.model}_search_results"] = self._save_csv(
                search.candidates, output / f"{search.model}_search_results.csv"
            )
            paths[f"{search.model}_best_params"] = self.storage.save_json(
                {
                    "model": search.model,
                    "best_params": search.best_params,
                    "selection_dataset": "train_time_series_cv",
                    "scoring": "f1_macro",
                },
                output / f"{search.model}_best_params.json",
            )
            paths[f"{search.model}_cv_summary"] = self.storage.save_json(
                {
                    "model": search.model,
                    "mean_cv_macro_f1": search.mean_cv_macro_f1,
                    "std_cv_macro_f1": search.std_cv_macro_f1,
                    "fold_scores": search.fold_scores,
                    "search_duration_seconds": search.duration_seconds,
                    "n_candidates": len(search.candidates),
                    "n_splits": self.settings.time_series_validation.n_splits,
                    "gap_rows": gap_rows,
                    "folds": fold_details,
                },
                output / f"{search.model}_cv_summary.json",
            )
        paths["best_params"] = self.storage.save_json(
            {search.model: search.best_params for search in searches},
            output / "best_params.json",
        )
        paths["cv_summary"] = self.storage.save_json(
            {
                "n_splits": self.settings.time_series_validation.n_splits,
                "gap_rows": gap_rows,
                "scoring": "f1_macro",
                "folds": fold_details,
                "models": {
                    search.model: {
                        "mean_cv_macro_f1": search.mean_cv_macro_f1,
                        "std_cv_macro_f1": search.std_cv_macro_f1,
                        "fold_scores": search.fold_scores,
                        "search_duration_seconds": search.duration_seconds,
                    }
                    for search in searches
                },
            },
            output / "cv_summary.json",
        )

        comparison_frame = pd.DataFrame(comparisons)
        paths["validation_comparison_csv"] = self._save_csv(
            comparison_frame, output / "validation_comparison.csv"
        )
        paths["validation_comparison_json"] = self.storage.save_json(
            {"models": comparisons}, output / "validation_comparison.json"
        )
        paths["validation_class_metrics"] = self.storage.save_json(
            {row["model"]: row["class_metrics"] for row in validation_results},
            output / "validation_class_metrics.json",
        )
        paths["validation_confusion_matrices"] = self.storage.save_json(
            {row["model"]: row["confusion_matrix"] for row in validation_results},
            output / "validation_confusion_matrices.json",
        )
        paths["best_tuned_model"] = self.storage.save_json(
            {
                "model": best["model"],
                "selection_dataset": "validation",
                "selection_metric": "macro_f1",
                "score": best["macro_f1"],
                "model_path": str(model_paths[str(best["model"])]),
            },
            output / "best_tuned_model.json",
        )
        paths["tuning_metadata"] = self.storage.save_json(
            {
                "train_dataset_path": str(self.dataset_paths["train"]),
                "validation_dataset_path": str(self.dataset_paths["validation"]),
                "test_dataset_path": str(self.dataset_paths["test"]),
                "test_dataset_exists": test_exists,
                "test_dataset_used": False,
                "validation_used_during_search": False,
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "features_count": len(feature_names),
                "features": feature_names,
                "n_splits": self.settings.time_series_validation.n_splits,
                "gap_rows": gap_rows,
                "target_horizon_hours": self.settings.target.horizon_hours,
                "scoring_config": self.settings.hyperparameter_search.scoring,
                "sklearn_scoring": sklearn_scoring_name(self.settings.hyperparameter_search.scoring),
                "n_iter": self.settings.hyperparameter_search.n_iter,
                "random_state": self.settings.hyperparameter_search.random_state,
                "fixed_model_parameters": {"lightgbm": {"subsample_freq": 1}},
                "folds": fold_details,
            },
            output / "tuning_metadata.json",
        )
        paths.update({f"model_{name}": path for name, path in model_paths.items()})
        return paths

    def run(self) -> TuningReport:
        """Execute all searches before loading Validation; never read Test."""
        LOGGER.info("Starting Block 4 tuning; search reads Train only and Test remains reserved")
        train, feature_names, test_exists = self._load_train()
        baseline_scores = self._load_block3_scores()
        gap_rows = derive_gap_rows(
            self.settings.target.horizon_hours, self.settings.market_data.timeframe
        )
        splitter = build_time_series_split(
            self.settings.time_series_validation.n_splits, gap_rows
        )
        fold_details = describe_folds(splitter, train["timestamp"])
        LOGGER.info(
            "Tuning inputs: train=%d features=%d n_splits=%d gap_rows=%d",
            len(train),
            len(feature_names),
            self.settings.time_series_validation.n_splits,
            gap_rows,
        )

        train_features = train[feature_names]
        train_target = train["target"]

        searches: list[ModelSearchResult] = []
        for model_name in self.tuner.model_names:
            searches.append(
                self.tuner.tune_model(model_name, train_features, train_target, splitter)
            )

        LOGGER.info("All Train-only searches finished; loading Validation now")
        validation = self._load_validation(train, feature_names)
        validation_features = validation[feature_names]
        validation_target = validation["target"].to_numpy()
        validation_results: list[dict[str, Any]] = []
        model_paths: dict[str, Path] = {}

        for search in searches:
            model = self._build_tuned_model(search.model, search.best_params)
            started = perf_counter()
            model.fit(train_features, train_target)
            training_time = perf_counter() - started
            predicted = model.predict(validation_features)
            probabilities = model.predict_proba(validation_features)
            if probabilities.shape != (len(validation), len(LABEL_ORDER)) or not np.isfinite(probabilities).all():
                raise ValueError(f"{model.name} returned invalid class probabilities")
            evaluation = evaluate_predictions(model.name, validation_target, predicted)
            evaluation["training_time_seconds"] = float(training_time)
            validation_results.append(evaluation)
            LOGGER.info(
                "%s Validation: accuracy=%.6f macro_precision=%.6f macro_recall=%.6f macro_f1=%.6f",
                model.name, evaluation["accuracy"], evaluation["macro_precision"],
                evaluation["macro_recall"], evaluation["macro_f1"],
            )

            model_path = self.settings.hyperparameter_search.models_dir / f"{model.name}_tuned.joblib"
            model.save(model_path)
            loaded = type(model).load(model_path)
            if not np.array_equal(predicted, loaded.predict(validation_features)):
                raise RuntimeError(f"{model.name} predictions changed after tuned model save/load")
            model_paths[model.name] = model_path

        comparisons = []
        search_by_name = {result.model: result for result in searches}
        for result in validation_results:
            name = str(result["model"])
            original = baseline_scores[name]
            comparisons.append({
                "model": name,
                "block3_validation_macro_f1": original,
                "tuned_validation_macro_f1": float(result["macro_f1"]),
                "tuned_validation_accuracy": float(result["accuracy"]),
                "tuned_validation_macro_precision": float(result["macro_precision"]),
                "tuned_validation_macro_recall": float(result["macro_recall"]),
                "final_training_time_seconds": float(result["training_time_seconds"]),
                "absolute_change": float(result["macro_f1"]) - original,
                "mean_cv_macro_f1": search_by_name[name].mean_cv_macro_f1,
                "std_cv_macro_f1": search_by_name[name].std_cv_macro_f1,
            })
        validation_results.sort(key=lambda row: float(row["macro_f1"]), reverse=True)
        comparisons.sort(key=lambda row: float(row["tuned_validation_macro_f1"]), reverse=True)
        best = select_best_model(validation_results, "macro_f1")
        artifact_paths = self._save_artifacts(
            searches, validation_results, comparisons, best, fold_details,
            len(train), len(validation), feature_names, gap_rows, test_exists, model_paths,
        )
        cv_results = [
            {
                "model": result.model,
                "mean_cv_macro_f1": result.mean_cv_macro_f1,
                "std_cv_macro_f1": result.std_cv_macro_f1,
                "fold_scores": result.fold_scores,
                "duration_seconds": result.duration_seconds,
                "best_params": result.best_params,
            }
            for result in searches
        ]
        LOGGER.info(
            "Block 4 complete; best tuned Validation model=%s macro_f1=%.6f",
            best["model"], best["macro_f1"],
        )
        return TuningReport(
            train_rows=len(train),
            validation_rows=len(validation),
            features_count=len(feature_names),
            n_splits=self.settings.time_series_validation.n_splits,
            gap_rows=gap_rows,
            cv_results=cv_results,
            validation_results=validation_results,
            best_model=str(best["model"]),
            best_score=float(best["macro_f1"]),
            artifact_paths=artifact_paths,
        )
