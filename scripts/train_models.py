"""Train and compare all configured Block 3 classifiers on Validation."""

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.training import ModelTrainingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = ModelTrainingPipeline(settings).run()
    print(f"Train rows: {report.train_rows}")
    print(f"Validation rows: {report.validation_rows}")
    print(f"Features: {report.features_count}")
    print("\nMODEL COMPARISON (Validation only)")
    print(f"{'Model':24} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Macro F1':>10} {'Time, s':>10}")
    for result in report.results:
        print(
            f"{result['model']:24} {result['accuracy']:10.4f} {result['macro_precision']:10.4f} "
            f"{result['macro_recall']:10.4f} {result['macro_f1']:10.4f} "
            f"{result['training_time_seconds']:10.3f}"
        )
    print(f"\nBest validation model: {report.best_model}")
    print(f"Validation Macro F1: {report.best_score:.6f}")
    print(f"Improvement over Dummy: {report.best_score - report.dummy_score:+.6f}")
    importance = json.loads(report.artifact_paths["feature_importance"].read_text(encoding="utf-8"))
    for model_name in ("catboost", "xgboost", "lightgbm"):
        print(f"\n{model_name} Top-10 features:")
        for index, row in enumerate(importance[model_name][:10], start=1):
            print(f"  {index:2}. {row['feature']}: {row['importance']:.6f}")
    print("\nTest dataset was not read or evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
