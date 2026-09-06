"""Data contracts for the backtesting subsystem."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


EQUITY_COLUMNS = [
    "timestamp", "cash", "btc_quantity", "btc_price", "position",
    "equity", "prediction", "executed_action", "execution_price",
]
TRADE_COLUMNS = [
    "entry_signal_time", "entry_time", "entry_price",
    "exit_signal_time", "exit_time", "exit_price",
    "btc_quantity", "entry_fee", "exit_fee", "total_fees",
    "holding_hours", "gross_pnl", "net_pnl", "return_pct", "exit_reason",
]


@dataclass(frozen=True)
class BacktestParameters:
    initial_cash: float
    commission_rate: float
    slippage_rate: float
    allow_short: bool = False
    leverage: float = 1.0
    force_close_at_end: bool = True

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 <= self.commission_rate < 1:
            raise ValueError("commission_rate must be in [0, 1)")
        if not 0 <= self.slippage_rate < 1:
            raise ValueError("slippage_rate must be in [0, 1)")
        if self.allow_short:
            raise ValueError("Short positions are disabled in Block 5")
        if self.leverage != 1.0:
            raise ValueError("Leverage must remain 1.0 in Block 5")


@dataclass
class BacktestResult:
    strategy_slug: str
    strategy_name: str
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]
    signal_counts: dict[str, int]
    buy_executions: int
    sell_executions: int


@dataclass(frozen=True)
class BacktestReport:
    period_start: str
    period_end: str
    validation_rows: int
    features_count: int
    results: list[dict[str, Any]]
    artifact_paths: dict[str, Path]
