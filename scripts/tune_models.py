"""Tune CatBoost, XGBoost, and LightGBM using Train-only time-series CV."""

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.tuning import HyperparameterTuningPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = HyperparameterTuningPipeline(settings).run()
    print(f"Train rows: {report.train_rows}")
    print(f"Validation rows: {report.validation_rows}")
    print(f"Features: {report.features_count}")
    print(f"TimeSeriesSplit: n_splits={report.n_splits}, gap={report.gap_rows}")
    print("\nTRAIN TIME-SERIES CV")
    print(f"{'Model':14} {'Mean Macro F1':>14} {'Std':>10} {'Time, s':>10}")
    for result in report.cv_results:
        print(
            f"{result['model']:14} {result['mean_cv_macro_f1']:14.6f} "
            f"{result['std_cv_macro_f1']:10.6f} {result['duration_seconds']:10.3f}"
        )
        print(f"  best_params={result['best_params']}")
        print(f"  fold_scores={[round(score, 6) for score in result['fold_scores']]}")
    print("\nVALIDATION: BLOCK 3 VS TUNED")
    comparison = json.loads(
        report.artifact_paths["validation_comparison_json"].read_text(encoding="utf-8")
    )["models"]
    print(f"{'Model':14} {'Block 3':>10} {'Tuned':>10} {'Difference':>12}")
    for result in comparison:
        print(
            f"{result['model']:14} {result['block3_validation_macro_f1']:10.6f} "
            f"{result['tuned_validation_macro_f1']:10.6f} {result['absolute_change']:+12.6f}"
        )

    print(f"\nBest tuned model: {report.best_model}")
    print(f"Validation Macro F1: {report.best_score:.6f}")
    print("Validation was not used inside hyperparameter search.")
    print("Test dataset was not read or evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
