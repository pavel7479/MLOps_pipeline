from pathlib import Path

import pandas as pd

from src.data.storage import ParquetStorage


def test_parquet_round_trip(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    storage = ParquetStorage()
    path = storage.save(valid_frame, tmp_path / "nested/data.parquet")
    assert path.exists()
    pd.testing.assert_frame_equal(valid_frame, storage.load(path))
