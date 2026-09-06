from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.config.settings import (
    CatBoostSettings, DummySettings, LightGBMSettings,
    LogisticRegressionSettings, XGBoostSettings,
)
from src.models import CatBoostModel, DummyModel, LightGBMModel, LogisticRegressionModel, XGBoostModel
from src.models.base import INVERSE_LABEL_MAPPING


def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    values, encoded = make_classification(
        n_samples=90, n_features=6, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=42,
    )
    features = pd.DataFrame(values, columns=[f"feature_{index}" for index in range(6)])
    target = pd.Series([INVERSE_LABEL_MAPPING[int(value)] for value in encoded])
    return features, target


@pytest.mark.parametrize(
    "model",
    [
        DummyModel(DummySettings()),
        LogisticRegressionModel(LogisticRegressionSettings(max_iter=200, random_state=42)),
        CatBoostModel(CatBoostSettings(iterations=5, depth=2, learning_rate=0.1, random_seed=42)),
        XGBoostModel(XGBoostSettings(n_estimators=5, max_depth=2, learning_rate=0.1, random_state=42)),
        LightGBMModel(LightGBMSettings(n_estimators=5, max_depth=2, learning_rate=0.1, random_state=42)),
    ],
    ids=["dummy", "logistic", "catboost", "xgboost", "lightgbm"],
)
def test_model_fit_probability_and_save_load(model, tmp_path: Path) -> None:
    features, target = classification_data()
    model.fit(features, target)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    assert len(predictions) == len(features)
    assert set(predictions).issubset({"BUY", "HOLD", "SELL"})
    assert probabilities.shape == (len(features), 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    path = model.save(tmp_path / f"{model.name}.joblib")
    loaded = type(model).load(path)
    np.testing.assert_array_equal(predictions, loaded.predict(features))
    np.testing.assert_allclose(probabilities, loaded.predict_proba(features))


def test_logistic_scaler_fits_train_only() -> None:
    features, target = classification_data()
    train_features = features.iloc[:60]
    validation_features = features.iloc[60:] + 1000
    model = LogisticRegressionModel(LogisticRegressionSettings(max_iter=200, random_state=42))
    model.fit(train_features, target.iloc[:60])
    scaler = model.estimator.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, train_features.mean().to_numpy())
    assert not np.allclose(scaler.mean_, pd.concat([train_features, validation_features]).mean().to_numpy())
    assert model.coefficient_report(list(features.columns))
