"""Financial metrics derived from an equity curve and completed trades."""

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365


def maximum_drawdown_pct(equity: pd.Series | list[float]) -> float:
    values = pd.Series(equity, dtype=float)
    if values.empty:
        return 0.0
    running_peak = values.cummax()
    drawdowns = values / running_peak - 1.0
    return float(drawdowns.min() * 100.0)


def hourly_sharpe_ratio(equity: pd.Series | list[float]) -> float:
    values = pd.Series(equity, dtype=float)
    returns = values.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    standard_deviation = float(returns.std(ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= np.finfo(float).eps:
        return 0.0
    return float(returns.mean() / standard_deviation * sqrt(HOURS_PER_YEAR))


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((trades["net_pnl"] > 0).sum() / len(trades))


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
    total_fees: float,
    long_periods: int,
    buy_executions: int,
    sell_executions: int,
) -> dict[str, Any]:
    if equity_curve.empty:
        raise ValueError("Equity curve cannot be empty")
    final_equity = float(equity_curve["equity"].iloc[-1])
    metric_equity = pd.Series(
        [float(initial_cash), *equity_curve["equity"].astype(float).tolist()]
    )
    completed = len(trades)
    profitable = int((trades["net_pnl"] > 0).sum()) if completed else 0
    losing = int((trades["net_pnl"] < 0).sum()) if completed else 0
    average_return = float(trades["return_pct"].mean()) if completed else 0.0
    average_holding = float(trades["holding_hours"].mean()) if completed else 0.0
    return {
        "initial_cash": float(initial_cash),
        "final_equity": final_equity,
        "total_return_pct": float((final_equity / initial_cash - 1.0) * 100.0),
        "max_drawdown_pct": maximum_drawdown_pct(metric_equity),
        "sharpe_ratio": hourly_sharpe_ratio(metric_equity),
        "number_of_trades": completed,
        "profitable_trades": profitable,
        "losing_trades": losing,
        "win_rate_pct": win_rate(trades) * 100.0,
        "average_trade_return_pct": average_return,
        "average_holding_hours": average_holding,
        "market_exposure_pct": float(long_periods / len(equity_curve) * 100.0),
        "total_fees": float(total_fees),
        "buy_executions": int(buy_executions),
        "sell_executions": int(sell_executions),
    }
