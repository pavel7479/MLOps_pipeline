"""Check Registry aliases and run prediction through the champion model URI."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
import pandas as pd

from src.config import load_settings
from src.mlops import MLflowTracker, ModelRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    tracker = MLflowTracker(settings.mlflow)
    tracker.check_health()
    registry = ModelRegistry(settings.mlflow)
    champion = registry.get_alias("champion")
    challenger = registry.get_alias("challenger")

    stem = f"{settings.market_data.symbol}_{settings.market_data.timeframe}"
    validation_path = settings.ml_dataset.output_dir / f"{stem}_validation.parquet"
    manifest_path = settings.ml_dataset.output_dir / "feature_manifest.json"
    feature_names = json.loads(manifest_path.read_text(encoding="utf-8"))["features"]
    validation = pd.read_parquet(validation_path)
    example = validation[feature_names].head(5)
    uri = f"models:/{settings.mlflow.registered_model_name}@champion"
    champion_model = mlflow.pyfunc.load_model(uri)
    predictions = champion_model.predict(example)

    print(f"Registered model: {settings.mlflow.registered_model_name}")
    print(
        f"Champion: version={champion.version}, run_id={champion.run_id}, "
        f"source={champion.source}"
    )
    print(
        f"Challenger: version={challenger.version}, run_id={challenger.run_id}, "
        f"source={challenger.source}"
    )
    print(f"Loaded URI: {uri}")
    print(f"Predictions: {list(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
