"""Safe Parquet persistence for raw and processed datasets."""

from pathlib import Path

import pandas as pd


class ParquetStorage:
    """Persist data atomically so interrupted writes keep prior data intact."""

    def save(self, data: pd.DataFrame, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        data.to_parquet(temporary, index=False)
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(Path(path))
