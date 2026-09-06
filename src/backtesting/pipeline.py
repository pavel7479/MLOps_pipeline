"""Load saved models, create Validation predictions, and run backtests."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import AppSettings
from src.data.models import timeframe_to_timedelta

from .engine import BacktestEngine
from .models import BacktestParameters, BacktestReport, BacktestResult
from .predictor import ArtifactPredictor
from .storage import BacktestArtifactWriter

LOGGER = logging.getLogger(__name__)


class BacktestingPipeline:
    """Backtest configured saved models on Validation without reading Test."""

    def __init__(
        self,
        settings: AppSettings,
        predictor: ArtifactPredictor | None = None,
        writer: BacktestArtifactWriter | None = None,
    ) -> None:
        self.settings = settings
        self.predictor = predictor or ArtifactPredictor()
        self.writer = writer or BacktestArtifactWriter()

    @property
    def dataset_paths(self) -> dict[str, Path]:
        stem = f"{self.settings.market_data.symbol}_{self.settings.market_data.timeframe}"
        root = self.settings.ml_dataset.output_dir
        return {
            "validation": root / f"{stem}_validation.parquet",
            "test": root / f"{stem}_test.parquet",
            "manifest": root / "feature_manifest.json",
        }

    def _load_validation(self) -> tuple[pd.DataFrame, list[str], bool]:
        paths = self.dataset_paths
        for key in ("validation", "manifest"):
            if not paths[key].is_file():
                raise FileNotFoundError(f"Required {key} artifact not found: {paths[key]}")
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        feature_names = manifest.get("features")
        if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
            raise ValueError("feature_manifest.json must contain a string list named features")
        validation = pd.read_parquet(paths["validation"])
        required = ["timestamp", "open", "close", *feature_names]
        missing = [column for column in required if column not in validation]
        if missing:
            raise ValueError(f"Validation is missing backtest columns: {missing}")
        if validation.empty:
            raise ValueError("Validation dataset cannot be empty")
        if validation["timestamp"].isna().any() or not validation["timestamp"].is_unique:
            raise ValueError("Validation timestamps must be non-null and unique")
        if not validation["timestamp"].is_monotonic_increasing:
            raise ValueError("Validation timestamps must be chronologically sorted")
        numeric = validation[["open", "close", *feature_names]]
        if (
            not all(pd.api.types.is_numeric_dtype(numeric[column]) for column in numeric)
            or numeric.isna().any().any()
            or not np.isfinite(numeric.to_numpy(dtype=float)).all()
        ):
            raise ValueError("Validation prices and features must be finite numeric values")
        if (validation[["open", "close"]] <= 0).any().any():
            raise ValueError("Validation prices must be positive")
        return validation, feature_names, paths["test"].is_file()

    def _parameters(self) -> BacktestParameters:
        configured = self.settings.backtesting
        return BacktestParameters(
            initial_cash=configured.initial_cash,
            commission_rate=configured.commission_rate,
            slippage_rate=configured.slippage_rate,
            allow_short=configured.allow_short,
            leverage=configured.leverage,
            force_close_at_end=configured.force_close_at_end,
        )

    @staticmethod
    def _result_summary(result: BacktestResult) -> dict[str, Any]:
        return {
            "strategy": result.strategy_name,
            "strategy_slug": result.strategy_slug,
            **result.metrics,
            "signal_counts": result.signal_counts,
        }

    def run(self) -> BacktestReport:
        LOGGER.info("Starting Validation-only backtest; Test will not be read")
        validation, feature_names, test_exists = self._load_validation()
        parameters = self._parameters()
        duration = timeframe_to_timedelta(self.settings.market_data.timeframe)
        engine = BacktestEngine(parameters, duration)
        market = validation[["timestamp", "open", "close"]].copy()
        LOGGER.info(
            "Backtest input: rows=%d features=%d period=%s..%s",
            len(validation), len(feature_names),
            pd.Timestamp(validation["timestamp"].iloc[0]).isoformat(),
            pd.Timestamp(validation["timestamp"].iloc[-1]).isoformat(),
        )

        buy_and_hold = engine.run_buy_and_hold(market)
        results = [buy_and_hold]
        model_metadata: list[dict[str, Any]] = []
        for strategy in self.settings.backtesting.strategies:
            prediction = self.predictor.predict(
                strategy.model_path, validation, feature_names
            )
            result = engine.run(
                market, prediction.predictions, strategy.slug, strategy.display_name
            )
            result.metrics["buy_and_hold_return_pct"] = buy_and_hold.metrics["total_return_pct"]
            result.metrics["excess_return_pct"] = (
                result.metrics["total_return_pct"] - buy_and_hold.metrics["total_return_pct"]
            )
            results.append(result)
            model_metadata.append({
                "strategy_slug": strategy.slug,
                "strategy_name": strategy.display_name,
                "model_path": str(strategy.model_path),
                "model_name": prediction.model_name,
                "model_version": strategy.model_version,
                "model_class": prediction.model_class,
                "model_parameters": prediction.parameters,
            })
            LOGGER.info(
                "%s: final_equity=%.2f return=%.4f%% drawdown=%.4f%% "
                "sharpe=%.4f trades=%d fees=%.2f signals=%s",
                strategy.display_name,
                result.metrics["final_equity"],
                result.metrics["total_return_pct"],
                result.metrics["max_drawdown_pct"],
                result.metrics["sharpe_ratio"],
                result.metrics["number_of_trades"],
                result.metrics["total_fees"],
                result.signal_counts,
            )
        buy_and_hold.metrics["buy_and_hold_return_pct"] = buy_and_hold.metrics["total_return_pct"]
        buy_and_hold.metrics["excess_return_pct"] = 0.0

        period_start = pd.Timestamp(validation["timestamp"].iloc[0]).isoformat()
        period_end = (
            pd.Timestamp(validation["timestamp"].iloc[-1]) + duration
        ).isoformat()
        metadata = {
            "backtest_period": {"start": period_start, "end": period_end},
            "validation_dataset_path": str(self.dataset_paths["validation"]),
            "test_dataset_path": str(self.dataset_paths["test"]),
            "test_dataset_exists": test_exists,
            "test_dataset_used": False,
            "validation_rows": len(validation),
            "features_count": len(feature_names),
            "features": feature_names,
            "initial_cash": parameters.initial_cash,
            "commission_rate": parameters.commission_rate,
            "slippage_rate": parameters.slippage_rate,
            "execution_rule": "prediction_after_close_t_executes_at_open_t_plus_1",
            "last_signal_execution": "ignored_without_next_candle",
            "force_close_rule": "last_available_candle_close",
            "position_rule": "all_in_FLAT_or_LONG",
            "allow_short": parameters.allow_short,
            "leverage": parameters.leverage,
            "force_close_at_end": parameters.force_close_at_end,
            "risk_free_rate": 0.0,
            "sharpe_annualization": "sqrt(24*365)",
            "models": model_metadata,
        }
        artifact_paths = self.writer.save(
            results, self.settings.backtesting.output_dir, metadata
        )
        LOGGER.info("Backtesting complete; artifacts=%s", self.settings.backtesting.output_dir)
        return BacktestReport(
            period_start=period_start,
            period_end=period_end,
            validation_rows=len(validation),
            features_count=len(feature_names),
            results=[self._result_summary(result) for result in results],
            artifact_paths=artifact_paths,
        )
