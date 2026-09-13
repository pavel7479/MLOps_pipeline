from pathlib import Path

import pandas as pd

import pytest

from src.data.storage import ParquetStorage, sha256_file


def test_parquet_round_trip(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    storage = ParquetStorage()
    path = storage.save(valid_frame, tmp_path / "nested/data.parquet")
    assert path.exists()
    pd.testing.assert_frame_equal(valid_frame, storage.load(path))


def test_sha256_is_stable_and_changes_with_content(
    tmp_path: Path, valid_frame: pd.DataFrame
) -> None:
    storage = ParquetStorage()
    path = storage.save(valid_frame, tmp_path / "data.parquet")
    first = sha256_file(path)
    assert first == sha256_file(path)
    storage.save(valid_frame.copy(), path)
    assert sha256_file(path) == first
    changed = valid_frame.copy()
    changed.loc[0, "close"] = 10.5
    storage.save(changed, path)
    assert sha256_file(path) != first


def test_failed_atomic_write_keeps_previous_file(
    tmp_path: Path, valid_frame: pd.DataFrame, monkeypatch
) -> None:
    storage = ParquetStorage()
    path = storage.save(valid_frame, tmp_path / "data.parquet")
    original_hash = sha256_file(path)

    def fail_write(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(RuntimeError, match="interrupted"):
        storage.save(valid_frame.iloc[:1], path)
    assert sha256_file(path) == original_hash
