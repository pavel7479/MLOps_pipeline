import numpy as np
import pandas as pd
import pytest

from src.training.validator import TrainingDataValidator


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"),
            "feature_a": np.arange(6, dtype=float),
            "target": ["SELL", "HOLD", "BUY", "SELL", "HOLD", "BUY"],
        }
    )
    validation = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=3, freq="h", tz="UTC"),
            "feature_a": np.arange(3, dtype=float),
            "target": ["SELL", "HOLD", "BUY"],
        }
    )
    return train, validation


def test_training_inputs_are_valid() -> None:
    TrainingDataValidator().validate(*frames(), ["feature_a"])


@pytest.mark.parametrize("forbidden", ["timestamp", "close", "future_return", "target"])
def test_manifest_rejects_leakage_fields(forbidden: str) -> None:
    with pytest.raises(ValueError, match="Forbidden"):
        TrainingDataValidator().validate(*frames(), [forbidden])


def test_nan_and_overlap_are_rejected() -> None:
    train, validation = frames()
    train.loc[0, "feature_a"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        TrainingDataValidator().validate(train, validation, ["feature_a"])
    train, validation = frames()
    validation.loc[0, "timestamp"] = train.loc[5, "timestamp"]
    with pytest.raises(ValueError, match="overlap"):
        TrainingDataValidator().validate(train, validation, ["feature_a"])
