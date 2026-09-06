import numpy as np
import pandas as pd
import pytest

from src.dataset.models import DatasetSplits
from src.dataset.validator import MLDatasetValidator


def valid_ml_frame(size: int = 9) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=size, freq="h", tz="UTC"),
            "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 5.0,
            "feature_a": np.arange(size, dtype=float), "future_return": 0.01, "target": "BUY",
        }
    )


def test_valid_ml_dataset_passes() -> None:
    MLDatasetValidator().validate_dataset(valid_ml_frame(), ["feature_a"])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_invalid_feature_values_fail(bad_value: float) -> None:
    frame = valid_ml_frame()
    frame.loc[2, "feature_a"] = bad_value
    with pytest.raises(ValueError, match="finite"):
        MLDatasetValidator().validate_dataset(frame, ["feature_a"])


def test_future_field_in_manifest_fails() -> None:
    with pytest.raises(ValueError, match="Future"):
        MLDatasetValidator().validate_dataset(valid_ml_frame(), ["future_return"])


def test_split_overlap_fails() -> None:
    frame = valid_ml_frame()
    splits = DatasetSplits(frame.iloc[:4], frame.iloc[3:6], frame.iloc[6:], 0)
    with pytest.raises(ValueError, match="overlap"):
        MLDatasetValidator().validate_splits(splits)
