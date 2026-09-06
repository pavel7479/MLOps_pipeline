"""Concrete baseline, linear, and gradient-boosting classifiers."""

from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config.settings import (
    CatBoostSettings,
    DummySettings,
    LightGBMSettings,
    LogisticRegressionSettings,
    XGBoostSettings,
)

from .base import BaseClassifier, INVERSE_LABEL_MAPPING


class DummyModel(BaseClassifier):
    """Most-frequent-class baseline."""

    name = "dummy"

    def __init__(self, settings: DummySettings) -> None:
        parameters = {"strategy": settings.strategy}
        super().__init__(DummyClassifier(**parameters), parameters)

    def coefficient_report(self, feature_names: list[str]) -> dict[str, list[dict[str, float | str]]]:
        return {}


class LogisticRegressionModel(BaseClassifier):
    """StandardScaler plus multinomial Logistic Regression baseline."""

    name = "logistic_regression"

    def __init__(self, settings: LogisticRegressionSettings) -> None:
        parameters: dict[str, Any] = {
            "max_iter": settings.max_iter,
            "random_state": settings.random_state,
            "class_weight": settings.class_weight,
        }
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**parameters)),
            ]
        )
        super().__init__(estimator, parameters)

    def coefficient_report(self, feature_names: list[str]) -> dict[str, list[dict[str, float | str]]]:
        classifier = self.estimator.named_steps["classifier"]
        report: dict[str, list[dict[str, float | str]]] = {}
        for encoded_class, coefficients in zip(classifier.classes_, classifier.coef_, strict=True):
            rows = [
                {"feature": feature, "coefficient": float(coefficient)}
                for feature, coefficient in zip(feature_names, coefficients, strict=True)
            ]
            report[INVERSE_LABEL_MAPPING[int(encoded_class)]] = sorted(
                rows, key=lambda row: abs(float(row["coefficient"])), reverse=True
            )
        return report


class CatBoostModel(BaseClassifier):
    """CatBoost multiclass classifier without scaling."""

    name = "catboost"

    def __init__(self, settings: CatBoostSettings) -> None:
        parameters = {
            "iterations": settings.iterations,
            "depth": settings.depth,
            "learning_rate": settings.learning_rate,
            "random_seed": settings.random_seed,
            "verbose": settings.verbose,
        }
        estimator = CatBoostClassifier(
            **parameters, loss_function="MultiClass", allow_writing_files=False, thread_count=1
        )
        super().__init__(estimator, parameters)

    def coefficient_report(self, feature_names: list[str]) -> dict[str, list[dict[str, float | str]]]:
        return {}


class XGBoostModel(BaseClassifier):
    """XGBoost multiclass classifier without scaling."""

    name = "xgboost"

    def __init__(self, settings: XGBoostSettings) -> None:
        parameters = {
            "n_estimators": settings.n_estimators,
            "max_depth": settings.max_depth,
            "learning_rate": settings.learning_rate,
            "random_state": settings.random_state,
        }
        estimator = XGBClassifier(
            **parameters, objective="multi:softprob", num_class=3, eval_metric="mlogloss", n_jobs=1
        )
        super().__init__(estimator, parameters)

    def coefficient_report(self, feature_names: list[str]) -> dict[str, list[dict[str, float | str]]]:
        return {}


class LightGBMModel(BaseClassifier):
    """LightGBM multiclass classifier without scaling."""

    name = "lightgbm"

    def __init__(self, settings: LightGBMSettings) -> None:
        parameters = {
            "n_estimators": settings.n_estimators,
            "max_depth": settings.max_depth,
            "learning_rate": settings.learning_rate,
            "random_state": settings.random_state,
        }
        estimator = LGBMClassifier(**parameters, objective="multiclass", num_class=3, verbosity=-1, n_jobs=1)
        super().__init__(estimator, parameters)

    def coefficient_report(self, feature_names: list[str]) -> dict[str, list[dict[str, float | str]]]:
        return {}
