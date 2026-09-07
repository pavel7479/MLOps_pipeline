"""MLflow model wrapper and reproducibility metadata helpers."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any

import mlflow
import pandas as pd

from src.models import BaseClassifier


class CryptoClassifierPyfunc(mlflow.pyfunc.PythonModel):
    """Expose the project classifier as a portable BUY/HOLD/SELL MLflow model."""

    def __init__(self, classifier: BaseClassifier, feature_names: list[str]) -> None:
        self.classifier = classifier
        self.feature_names = list(feature_names)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> Any:
        del context, params
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self.feature_names)
        missing = [name for name in self.feature_names if name not in model_input]
        if missing:
            raise ValueError(f"MLflow model input is missing features: {missing}")
        return self.classifier.predict(model_input[self.feature_names])


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 fingerprint without loading a whole dataset."""
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repository: str | Path) -> dict[str, str]:
    """Best-effort branch and commit metadata; Git availability is non-critical."""

    def read(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=Path(repository),
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            value = completed.stdout.strip()
            return value or "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    return {
        "git_branch": read("branch", "--show-current"),
        "git_commit": read("rev-parse", "HEAD"),
    }


def dataframe_period(frame: pd.DataFrame) -> dict[str, str]:
    """Describe the inclusive timestamp range of a chronological split."""
    if "timestamp" not in frame or frame.empty:
        return {"start": "unknown", "end": "unknown"}
    return {
        "start": pd.Timestamp(frame["timestamp"].iloc[0]).isoformat(),
        "end": pd.Timestamp(frame["timestamp"].iloc[-1]).isoformat(),
    }
