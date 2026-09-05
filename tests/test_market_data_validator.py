import pandas as pd

from src.data.validator import MarketDataValidator


def test_valid_data_passes(valid_frame: pd.DataFrame) -> None:
    assert MarketDataValidator().validate(valid_frame, "1h").is_valid


def test_detects_duplicates_invalid_values_and_null(valid_frame: pd.DataFrame) -> None:
    frame = pd.concat([valid_frame, valid_frame.iloc[[0]]], ignore_index=True)
    frame.loc[1, "high"] = 1
    frame.loc[2, "open"] = -1
    frame.loc[3, "volume"] = None
    result = MarketDataValidator().validate(frame, "1h")
    assert result.duplicate_timestamps == 2
    assert result.invalid_rows == 3
    assert not result.is_valid


def test_detects_missing_hour(valid_frame: pd.DataFrame) -> None:
    frame = valid_frame.drop(index=1)
    result = MarketDataValidator().validate(frame, "1h")
    assert result.missing_intervals == 1
