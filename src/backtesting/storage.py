"""Atomic persistence for backtest results."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset.storage import MLDatasetStorage

from .models import BacktestResult


class BacktestArtifactWriter:
    def __init__(self) -> None:
        self.storage = MLDatasetStorage()

    @staticmethod
    def _save_csv(frame: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        return path

    def save(
        self,
        results: list[BacktestResult],
        output_dir: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        comparison_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for result in results:
            equity_path = output_dir / f"equity_curve_{result.strategy_slug}.csv"
            trades_path = output_dir / f"trades_{result.strategy_slug}.csv"
            paths[f"equity_{result.strategy_slug}"] = self._save_csv(
                result.equity_curve, equity_path
            )
            paths[f"trades_{result.strategy_slug}"] = self._save_csv(
                result.trades, trades_path
            )
            row = {
                "strategy": result.strategy_name,
                "strategy_slug": result.strategy_slug,
                **result.metrics,
            }
            comparison_rows.append(row)
            summaries.append({
                **row,
                "signal_counts": result.signal_counts,
            })
        paths["strategy_comparison"] = self._save_csv(
            pd.DataFrame(comparison_rows), output_dir / "strategy_comparison.csv"
        )
        paths["backtest_summary"] = self.storage.save_json(
            {"strategies": summaries}, output_dir / "backtest_summary.json"
        )
        paths["backtest_metadata"] = self.storage.save_json(
            metadata, output_dir / "backtest_metadata.json"
        )
        return paths
