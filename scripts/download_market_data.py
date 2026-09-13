"""Run the configured market data ingestion pipeline."""

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.data import MarketDataPipeline, MultiAssetMarketDataPipeline


def _print_legacy_report(report) -> None:
    for label, value in (
        ("Downloaded rows", report.downloaded_rows),
        ("Duplicate timestamps", report.duplicate_timestamps),
        ("Invalid rows", report.invalid_rows),
        ("Missing intervals", report.missing_intervals),
        ("Final rows", report.final_rows),
        ("Start timestamp", report.start_timestamp),
        ("End timestamp", report.end_timestamp),
        ("Raw dataset", report.raw_path),
        ("Processed dataset", report.processed_path),
    ):
        print(f"{label}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--all",
        action="store_true",
        help="download every symbol x interval pair from market_data config",
    )
    parser.add_argument("--symbol", help="download one configured symbol")
    parser.add_argument("--interval", help="download one configured interval")
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=getattr(logging, settings.logging.level, logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.all and (args.symbol or args.interval):
        parser.error("--all cannot be combined with --symbol or --interval")
    if bool(args.symbol) != bool(args.interval):
        parser.error("--symbol and --interval must be supplied together")

    if not args.all and not args.symbol:
        _print_legacy_report(MarketDataPipeline(settings).run())
        return 0

    if args.symbol:
        symbol = args.symbol.strip().upper()
        interval = args.interval.strip()
        if symbol not in settings.market_data.configured_symbols:
            parser.error(
                f"unknown symbol {symbol}; configured: "
                f"{', '.join(settings.market_data.configured_symbols)}"
            )
        if interval not in settings.market_data.configured_intervals:
            parser.error(
                f"unknown interval {interval}; configured: "
                f"{', '.join(settings.market_data.configured_intervals)}"
            )
        symbols, intervals = (symbol,), (interval,)
    else:
        symbols = settings.market_data.configured_symbols
        intervals = settings.market_data.configured_intervals

    result = MultiAssetMarketDataPipeline(settings).run(symbols, intervals)
    for item in result.datasets:
        print(
            f"{item.symbol} {item.interval} PASS: rows={item.row_count}, "
            f"from={item.start_timestamp}, to={item.end_timestamp}, "
            f"duplicates={item.duplicates}, gaps={item.missing_intervals}, "
            f"path={item.file_path}"
        )
    print(
        f"Dataset summary: {result.successful_datasets}/{result.requested_datasets} PASS"
    )
    for failure in result.failures:
        print(f"FAIL {failure.symbol} {failure.interval}: {failure.error}")
    print(f"Manifest: {result.manifest_path}")
    return 0 if result.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
