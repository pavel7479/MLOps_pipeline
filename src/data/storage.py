"""Safe Parquet persistence for raw and processed datasets."""

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest of an on-disk artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
