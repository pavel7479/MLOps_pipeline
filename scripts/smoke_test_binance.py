"""Small real-network smoke test for Binance public market-data access."""

import argparse
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.data.binance_provider import BinanceMarketDataProvider
from src.data.validator import MarketDataValidator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--intervals", nargs="+", default=["1h", "1d"])
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be positive")

    settings = load_settings(args.config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    provider = BinanceMarketDataProvider(
        settings.http.timeout_seconds,
        settings.http.max_retries,
        settings.http.retry_delay_seconds,
    )
    validator = MarketDataValidator()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    failures = 0

    for symbol in args.symbols:
        for interval in args.intervals:
            try:
                frame = provider.fetch_ohlcv(symbol.upper(), interval, start, end)
                result = validator.validate(frame, interval)
                fatal = [
                    error
                    for error in result.errors
                    if not error.startswith("Missing intervals:")
                ]
                if frame.empty or fatal:
                    raise ValueError("; ".join(fatal) or "empty response")
                print(
                    f"PASS {symbol.upper()} {interval}: rows={len(frame)}, "
                    f"first={frame['timestamp'].min()}, last={frame['timestamp'].max()}"
                )
            except Exception as exc:
                failures += 1
                print(f"FAIL {symbol.upper()} {interval}: {exc}")

    print(f"Binance smoke summary: failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
