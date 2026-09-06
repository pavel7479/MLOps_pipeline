"""Model-agnostic next-candle-open backtesting engine."""

from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import calculate_metrics
from .models import (
    BacktestParameters, BacktestResult, EQUITY_COLUMNS, TRADE_COLUMNS,
)
from .portfolio import BuyExecution, Portfolio, SellExecution

ALLOWED_SIGNALS = {"BUY", "HOLD", "SELL"}


class BacktestEngine:
    """Simulate fixed all-in LONG/FLAT rules from prepared predictions."""

    def __init__(self, parameters: BacktestParameters, candle_duration: pd.Timedelta) -> None:
        parameters.validate()
        if candle_duration <= pd.Timedelta(0):
            raise ValueError("candle_duration must be positive")
        self.parameters = parameters
        self.candle_duration = candle_duration

    @staticmethod
    def _validate_inputs(market: pd.DataFrame, predictions: Sequence[str]) -> pd.Series:
        required = ["timestamp", "open", "close"]
        missing = [column for column in required if column not in market]
        if missing:
            raise ValueError(f"Backtest market data is missing columns: {missing}")
        if market.empty:
            raise ValueError("Backtest market data cannot be empty")
        if len(predictions) != len(market):
            raise ValueError("Predictions count must equal market row count")
        if market["timestamp"].isna().any() or not market["timestamp"].is_unique:
            raise ValueError("Backtest timestamps must be non-null and unique")
        if not market["timestamp"].is_monotonic_increasing:
            raise ValueError("Backtest market data must be chronologically sorted")
        prices = market[["open", "close"]]
        if (
            not all(pd.api.types.is_numeric_dtype(prices[column]) for column in prices)
            or prices.isna().any().any()
            or not np.isfinite(prices.to_numpy(dtype=float)).all()
            or (prices <= 0).any().any()
        ):
            raise ValueError("Backtest open/close prices must be finite and positive")
        signals = pd.Series(predictions, index=market.index, dtype="string")
        invalid = set(signals.dropna().unique()) - ALLOWED_SIGNALS
        if signals.isna().any() or invalid:
            raise ValueError(f"Predictions must contain only BUY/HOLD/SELL; invalid={sorted(invalid)}")
        return signals

    @staticmethod
    def _trade_row(
        entry: dict[str, Any],
        sell: SellExecution,
        exit_signal_time: Any,
        exit_time: pd.Timestamp,
        exit_reason: str,
    ) -> dict[str, Any]:
        gross_pnl = sell.btc_quantity * (sell.execution_price - entry["entry_price"])
        total_fees = entry["entry_fee"] + sell.fee
        net_pnl = gross_pnl - total_fees
        return {
            "entry_signal_time": entry["entry_signal_time"],
            "entry_time": entry["entry_time"],
            "entry_price": entry["entry_price"],
            "exit_signal_time": exit_signal_time,
            "exit_time": exit_time,
            "exit_price": sell.execution_price,
            "btc_quantity": sell.btc_quantity,
            "entry_fee": entry["entry_fee"],
            "exit_fee": sell.fee,
            "total_fees": total_fees,
            "holding_hours": float((exit_time - entry["entry_time"]) / pd.Timedelta(1, unit="h")),
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "return_pct": float(net_pnl / entry["total_cost"] * 100.0),
            "exit_reason": exit_reason,
        }

    def run(
        self,
        market: pd.DataFrame,
        predictions: Sequence[str],
        strategy_slug: str,
        strategy_name: str,
    ) -> BacktestResult:
        """Run signals made at close t and execute them at open t+1."""
        signals = self._validate_inputs(market, predictions)
        portfolio = Portfolio(self.parameters)
        curve_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        active_entry: dict[str, Any] | None = None
        total_fees = 0.0
        buy_executions = 0
        sell_executions = 0
        long_periods = 0

        for offset in range(len(market)):
            row = market.iloc[offset]
            executed_action = "NONE"
            execution_price: float | None = None
            if offset > 0:
                previous_signal = str(signals.iloc[offset - 1])
                signal_time = pd.Timestamp(market["timestamp"].iloc[offset - 1])
                execution_time = pd.Timestamp(row["timestamp"])
                if previous_signal == "BUY" and portfolio.position == "FLAT":
                    buy: BuyExecution = portfolio.buy(float(row["open"]))
                    total_fees += buy.fee
                    buy_executions += 1
                    executed_action = "BUY"
                    execution_price = buy.execution_price
                    active_entry = {
                        "entry_signal_time": signal_time,
                        "entry_time": execution_time,
                        "entry_price": buy.execution_price,
                        "entry_fee": buy.fee,
                        "total_cost": buy.total_cost,
                    }
                elif previous_signal == "SELL" and portfolio.position == "LONG":
                    sell = portfolio.sell(float(row["open"]))
                    total_fees += sell.fee
                    sell_executions += 1
                    executed_action = "SELL"
                    execution_price = sell.execution_price
                    if active_entry is None:
                        raise RuntimeError("LONG portfolio has no entry accounting")
                    trade_rows.append(
                        self._trade_row(
                            active_entry, sell, signal_time, execution_time, "sell_signal"
                        )
                    )
                    active_entry = None

            if portfolio.position == "LONG":
                long_periods += 1
            curve_rows.append({
                "timestamp": pd.Timestamp(row["timestamp"]),
                "cash": portfolio.cash,
                "btc_quantity": portfolio.btc_quantity,
                "btc_price": float(row["close"]),
                "position": portfolio.position,
                "equity": portfolio.equity(float(row["close"])),
                "prediction": str(signals.iloc[offset]),
                "executed_action": executed_action,
                "execution_price": execution_price,
            })

        if portfolio.position == "LONG" and self.parameters.force_close_at_end:
            last = market.iloc[-1]
            close_time = pd.Timestamp(last["timestamp"]) + self.candle_duration
            sell = portfolio.sell(float(last["close"]))
            total_fees += sell.fee
            sell_executions += 1
            if active_entry is None:
                raise RuntimeError("LONG portfolio has no entry accounting")
            trade_rows.append(
                self._trade_row(active_entry, sell, pd.NaT, close_time, "end_of_backtest")
            )
            last_curve = curve_rows[-1]
            prior_action = str(last_curve["executed_action"])
            last_curve.update({
                "cash": portfolio.cash,
                "btc_quantity": 0.0,
                "position": "FLAT",
                "equity": portfolio.cash,
                "executed_action": (
                    "FORCE_SELL" if prior_action == "NONE" else f"{prior_action}+FORCE_SELL"
                ),
                "execution_price": sell.execution_price,
            })

        equity_curve = pd.DataFrame(curve_rows, columns=EQUITY_COLUMNS)
        trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
        metrics = calculate_metrics(
            equity_curve, trades, self.parameters.initial_cash, total_fees,
            long_periods, buy_executions, sell_executions,
        )
        counts = Counter(str(signal) for signal in signals)
        signal_counts = {signal: int(counts.get(signal, 0)) for signal in ("BUY", "HOLD", "SELL")}
        return BacktestResult(
            strategy_slug, strategy_name, equity_curve, trades, metrics,
            signal_counts, buy_executions, sell_executions,
        )

    def run_buy_and_hold(
        self,
        market: pd.DataFrame,
        strategy_slug: str = "buy_and_hold",
        strategy_name: str = "Buy & Hold",
    ) -> BacktestResult:
        """Buy at the first Validation open and sell at its final close."""
        self._validate_inputs(market, ["HOLD"] * len(market))
        portfolio = Portfolio(self.parameters)
        first = market.iloc[0]
        buy = portfolio.buy(float(first["open"]))
        total_fees = buy.fee
        curve_rows: list[dict[str, Any]] = []
        for offset in range(len(market)):
            row = market.iloc[offset]
            curve_rows.append({
                "timestamp": pd.Timestamp(row["timestamp"]),
                "cash": portfolio.cash,
                "btc_quantity": portfolio.btc_quantity,
                "btc_price": float(row["close"]),
                "position": "LONG",
                "equity": portfolio.equity(float(row["close"])),
                "prediction": "BUY_AND_HOLD",
                "executed_action": "BUY" if offset == 0 else "NONE",
                "execution_price": buy.execution_price if offset == 0 else None,
            })
        last = market.iloc[-1]
        exit_time = pd.Timestamp(last["timestamp"]) + self.candle_duration
        sell = portfolio.sell(float(last["close"]))
        total_fees += sell.fee
        trade = self._trade_row(
            {
                "entry_signal_time": pd.NaT,
                "entry_time": pd.Timestamp(first["timestamp"]),
                "entry_price": buy.execution_price,
                "entry_fee": buy.fee,
                "total_cost": buy.total_cost,
            },
            sell,
            pd.NaT,
            exit_time,
            "end_of_backtest",
        )
        curve_rows[-1].update({
            "cash": portfolio.cash,
            "btc_quantity": 0.0,
            "position": "FLAT",
            "equity": portfolio.cash,
            "executed_action": (
                "FORCE_SELL" if len(market) > 1 else "BUY+FORCE_SELL"
            ),
            "execution_price": sell.execution_price,
        })
        equity_curve = pd.DataFrame(curve_rows, columns=EQUITY_COLUMNS)
        trades = pd.DataFrame([trade], columns=TRADE_COLUMNS)
        metrics = calculate_metrics(
            equity_curve, trades, self.parameters.initial_cash, total_fees,
            len(market), 1, 1,
        )
        return BacktestResult(
            strategy_slug, strategy_name, equity_curve, trades, metrics,
            {"BUY": 0, "HOLD": 0, "SELL": 0}, 1, 1,
        )
