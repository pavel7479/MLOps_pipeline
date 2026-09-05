"""Run the configured market data ingestion pipeline."""

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.data import MarketDataPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=getattr(logging, settings.logging.level, logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = MarketDataPipeline(settings).run()
    for label, value in (
        ("Downloaded rows", report.downloaded_rows), ("Duplicate timestamps", report.duplicate_timestamps),
        ("Invalid rows", report.invalid_rows), ("Missing intervals", report.missing_intervals),
        ("Final rows", report.final_rows), ("Start timestamp", report.start_timestamp),
        ("End timestamp", report.end_timestamp), ("Raw dataset", report.raw_path),
        ("Processed dataset", report.processed_path),
    ):
        print(f"{label}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
