"""Persistence helpers for ML datasets and reproducibility artifacts."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.storage import ParquetStorage


class MLDatasetStorage:
    """Atomically persist Parquet datasets and JSON artifacts."""

    def __init__(self) -> None:
        self.parquet = ParquetStorage()

    def save_frame(self, data: pd.DataFrame, path: Path) -> Path:
        """Save a dataframe using the ingestion layer's atomic Parquet writer."""
        return self.parquet.save(data, path)

    def save_json(self, payload: dict[str, Any], path: Path) -> Path:
        """Write readable deterministic JSON through an atomic replacement."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path
