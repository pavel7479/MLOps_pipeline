"""Build the configured leakage-safe ML dataset and chronological splits."""

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.dataset import MLDatasetPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = MLDatasetPipeline(settings).run()
    print(f"Input rows: {report.input_rows}")
    print(f"Final rows: {report.final_rows}")
    print(f"Feature count: {report.feature_count}")
    print("Target distribution:")
    for label, values in report.target_distribution.items():
        print(f"  {label}: {values['count']} ({values['percentage']:.2f}%)")
    print(f"Train rows/range: {report.train_rows} | {report.train_range[0]} .. {report.train_range[1]}")
    print(f"Validation rows/range: {report.validation_rows} | {report.validation_range[0]} .. {report.validation_range[1]}")
    print(f"Test rows/range: {report.test_rows} | {report.test_range[0]} .. {report.test_range[1]}")
    print(f"Purged boundary rows: {report.purged_rows}")
    print(f"Dataset saved: {report.dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
