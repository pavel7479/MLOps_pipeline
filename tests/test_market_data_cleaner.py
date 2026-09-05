import pandas as pd

from src.data.cleaner import MarketDataCleaner


def test_cleaner_removes_bad_rows_duplicates_and_sorts(valid_frame: pd.DataFrame) -> None:
    duplicate = valid_frame.iloc[[0]].copy()
    invalid = valid_frame.iloc[[1]].copy()
    invalid["close"] = -2
    dirty = pd.concat([valid_frame.iloc[::-1], duplicate, invalid], ignore_index=True)
    cleaned = MarketDataCleaner().clean(dirty)
    assert len(cleaned) == 3
    assert cleaned["timestamp"].is_monotonic_increasing
    assert cleaned["timestamp"].is_unique
    assert str(cleaned["timestamp"].dtype) == "datetime64[ns, UTC]"
    pd.testing.assert_frame_equal(cleaned, MarketDataCleaner().clean(cleaned))
