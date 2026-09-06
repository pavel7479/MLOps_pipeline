import pandas as pd

from src.config.settings import SplitSettings
from src.dataset.splitter import ChronologicalDatasetSplitter


def test_chronological_ratios_no_overlap_and_purge() -> None:
    data = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"), "row_id": range(100)}
    )
    splits = ChronologicalDatasetSplitter(SplitSettings()).split(data, purge_periods=3)
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (67, 12, 15)
    assert splits.purged_rows == 6
    assert splits.train["timestamp"].is_monotonic_increasing
    assert splits.train["timestamp"].max() < splits.validation["timestamp"].min()
    assert splits.validation["timestamp"].max() < splits.test["timestamp"].min()
    assert set(splits.train["timestamp"]).isdisjoint(splits.validation["timestamp"])
    assert splits.train["row_id"].iloc[-1] + 3 < splits.validation["row_id"].iloc[0]
    assert splits.validation["row_id"].iloc[-1] + 3 < splits.test["row_id"].iloc[0]
