"""Build the versioned Train-only PSI reference artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_settings
from src.monitoring.reference import build_reference, write_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--train", type=Path, default=Path("data/ml/BTCUSDT_1h_train.parquet")
    )
    parser.add_argument(
        "--feature-manifest", type=Path, default=Path("data/ml/feature_manifest.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("monitoring/reference/BTCUSDT_1h_train_reference.json"),
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    feature_names = tuple(manifest["features"])
    train = pd.read_parquet(args.train)
    reference = build_reference(
        train,
        feature_names,
        symbol=settings.market_data.symbol,
        timeframe=settings.market_data.timeframe,
        train_path=args.train,
        manifest_path=args.feature_manifest,
    )
    write_reference(reference, args.output)
    print(
        f"Monitoring reference written: {args.output} "
        f"({len(train)} Train rows, {len(feature_names)} features)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
