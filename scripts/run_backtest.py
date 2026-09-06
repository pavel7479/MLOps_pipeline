"""Run Validation-only virtual trading for saved Block 3 and 4 models."""

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting import BacktestingPipeline
from src.config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = BacktestingPipeline(settings).run()
    configured = settings.backtesting
    print("BACKTEST PERIOD")
    print(f"{report.period_start} .. {report.period_end}")
    print(f"Validation rows: {report.validation_rows}")
    print(f"Features: {report.features_count}")
    print(f"Initial capital: {configured.initial_cash:,.2f} USDT")
    print(f"Commission: {configured.commission_rate:.4%}")
    print(f"Slippage: {configured.slippage_rate:.4%}")
    print("Execution: signal after close t -> open t+1")
    print("\nSTRATEGY COMPARISON")
    print(
        f"{'Strategy':24} {'Final Equity':>14} {'Return':>10} {'Max DD':>10} "
        f"{'Sharpe':>9} {'Trades':>8} {'Win Rate':>10} {'Fees':>11}"
    )
    for result in report.results:
        print(
            f"{result['strategy']:24} {result['final_equity']:14.2f} "
            f"{result['total_return_pct']:9.3f}% {result['max_drawdown_pct']:9.3f}% "
            f"{result['sharpe_ratio']:9.3f} {result['number_of_trades']:8d} "
            f"{result['win_rate_pct']:9.2f}% {result['total_fees']:11.2f}"
        )
        if result["strategy_slug"] != "buy_and_hold":
            print(
                f"  excess vs Buy & Hold: {result['excess_return_pct']:+.3f} pp; "
                f"signals={result['signal_counts']}; "
                f"executions=BUY {result['buy_executions']}, SELL {result['sell_executions']}"
            )
            print(
                f"  profitable={result['profitable_trades']}, losing={result['losing_trades']}, "
                f"avg trade={result['average_trade_return_pct']:.3f}%, "
                f"avg holding={result['average_holding_hours']:.2f}h, "
                f"exposure={result['market_exposure_pct']:.2f}%"
            )
    print("\nThis is a historical Validation simulation, not real trading.")
    print("Test dataset was not read or used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
