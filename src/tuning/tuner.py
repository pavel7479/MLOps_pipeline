"""Randomized hyperparameter search on Train only."""

from dataclasses import asdict
import logging
from time import perf_counter

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.config.settings import AppSettings
from src.models import CatBoostModel, LABEL_MAPPING, LightGBMModel, XGBoostModel

from .models import ModelSearchResult

LOGGER = logging.getLogger(__name__)


def sklearn_scoring_name(configured_name: str) -> str:
    """Map the project metric name to sklearn's explicit multiclass scorer."""
    if configured_name != "macro_f1":
        raise ValueError(f"Unsupported hyperparameter-search scoring: {configured_name}")
    return "f1_macro"


class HyperparameterTuner:
    """Tune only the three gradient-boosting estimators."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def _estimators_and_spaces(self) -> dict[str, tuple[object, dict[str, tuple]]]:
        spaces = self.settings.search_spaces
        lightgbm_estimator = LightGBMModel(self.settings.models.lightgbm).estimator
        lightgbm_estimator.set_params(subsample_freq=1)
        return {
            "catboost": (CatBoostModel(self.settings.models.catboost).estimator, asdict(spaces.catboost)),
            "xgboost": (XGBoostModel(self.settings.models.xgboost).estimator, asdict(spaces.xgboost)),
            "lightgbm": (lightgbm_estimator, asdict(spaces.lightgbm)),
        }

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(self._estimators_and_spaces())

    def tune_model(
        self,
        model_name: str,
        features: pd.DataFrame,
        target: pd.Series,
        splitter: TimeSeriesSplit,
    ) -> ModelSearchResult:
        """Run deterministic RandomizedSearchCV and retain all fold results."""
        try:
            estimator, parameter_space = self._estimators_and_spaces()[model_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported tuned model: {model_name}") from exc
        encoded_target = target.map(LABEL_MAPPING)
        if encoded_target.isna().any():
            raise ValueError("Train target contains an unsupported label")
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=parameter_space,
            n_iter=self.settings.hyperparameter_search.n_iter,
            scoring=sklearn_scoring_name(self.settings.hyperparameter_search.scoring),
            cv=splitter,
            refit=False,
            random_state=self.settings.hyperparameter_search.random_state,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        LOGGER.info(
            "Searching %s: n_iter=%d folds=%d scoring=f1_macro",
            model_name,
            self.settings.hyperparameter_search.n_iter,
            self.settings.time_series_validation.n_splits,
        )
        started = perf_counter()
        search.fit(features, encoded_target.astype(int))
        duration = perf_counter() - started
        best_index = int(search.best_index_)
        raw_results = pd.DataFrame(search.cv_results_)
        split_columns = [
            f"split{index}_test_score"
            for index in range(self.settings.time_series_validation.n_splits)
        ]
        fold_scores = [float(raw_results.loc[best_index, column]) for column in split_columns]
        columns = [
            "rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time",
            "mean_score_time", "params", *split_columns,
        ]
        parameter_columns = sorted(column for column in raw_results if column.startswith("param_"))
        candidates = raw_results[[*columns, *parameter_columns]].sort_values(
            "rank_test_score", kind="stable"
        )
        result = ModelSearchResult(
            model=model_name,
            best_params={key: value.item() if hasattr(value, "item") else value for key, value in search.best_params_.items()},
            mean_cv_macro_f1=float(search.best_score_),
            std_cv_macro_f1=float(raw_results.loc[best_index, "std_test_score"]),
            fold_scores=fold_scores,
            duration_seconds=float(duration),
            candidates=candidates,
        )
        LOGGER.info(
            "%s search complete in %.3fs; CV macro_f1=%.6f +/- %.6f",
            model_name, duration, result.mean_cv_macro_f1, result.std_cv_macro_f1,
        )
        LOGGER.info("%s best parameters: %s", model_name, result.best_params)
        return result
