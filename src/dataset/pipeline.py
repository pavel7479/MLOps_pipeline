"""End-to-end preparation of a leakage-safe ML dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import AppSettings
from src.data.models import MARKET_COLUMNS, timeframe_to_timedelta
from src.data.validator import MarketDataValidator
from src.features.feature_engineer import FeatureEngineer
from src.features.target import TARGET_CLASSES, TargetBuilder

from .models import MLDatasetReport
from .splitter import ChronologicalDatasetSplitter
from .storage import MLDatasetStorage
from .validator import MLDatasetValidator

LOGGER = logging.getLogger(__name__)


class MLDatasetPipeline:
    """Build, validate, split, and persist a reproducible ML dataset."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.feature_engineer = FeatureEngineer(settings.features)
        self.target_builder = TargetBuilder(settings.target)
        self.splitter = ChronologicalDatasetSplitter(settings.split)
        self.validator = MLDatasetValidator()
        self.storage = MLDatasetStorage()

    @property
    def input_path(self) -> Path:
        market = self.settings.market_data
        return self.settings.storage.processed_dir / f"{market.symbol}_{market.timeframe}.parquet"

    def _horizon_periods(self) -> int:
        timeframe = timeframe_to_timedelta(self.settings.market_data.timeframe)
        horizon = pd.Timedelta(self.settings.target.horizon_hours, unit="h")
        periods = horizon / timeframe
        if periods < 1 or not float(periods).is_integer():
            raise ValueError("Target horizon must be an exact multiple of the market timeframe")
        return int(periods)

    @staticmethod
    def _time_range(data: pd.DataFrame) -> tuple[object, object]:
        return (data["timestamp"].iloc[0].to_pydatetime(), data["timestamp"].iloc[-1].to_pydatetime())

    def _target_distribution(self, data: pd.DataFrame) -> dict[str, dict[str, float | int]]:
        counts = data["target"].value_counts()
        total = len(data)
        distribution: dict[str, dict[str, float | int]] = {}
        for target_class in TARGET_CLASSES:
            count = int(counts.get(target_class, 0))
            percentage = count / total * 100
            distribution[target_class] = {"count": count, "percentage": percentage}
            if count / total < self.settings.target.rare_class_warning_threshold:
                LOGGER.warning("Rare target class %s: %.2f%%", target_class, percentage)
        return distribution

    def run(self) -> MLDatasetReport:
        """Execute all preparation steps and return a structured quality report."""
        path = self.input_path
        LOGGER.info("Loading processed OHLCV dataset: %s", path)
        if not path.is_file():
            raise FileNotFoundError(f"Processed OHLCV dataset not found: {path}")
        source = self.storage.parquet.load(path)
        input_rows = len(source)
        if source.empty:
            raise ValueError("Processed OHLCV dataset is empty")
        input_validation = MarketDataValidator().validate(source, self.settings.market_data.timeframe)
        if not input_validation.is_valid:
            raise ValueError(f"Processed OHLCV validation failed: {'; '.join(input_validation.errors)}")

        LOGGER.info("Calculating features for %d input rows", input_rows)
        featured = self.feature_engineer.transform(source[MARKET_COLUMNS])
        feature_names = self.feature_engineer.feature_names
        finite_features = np.isfinite(featured[feature_names].to_numpy(dtype=float)).all(axis=1)
        feature_ready = featured[feature_names].notna().all(axis=1) & finite_features
        rows_removed_for_features = input_rows - int(feature_ready.sum())
        LOGGER.info("Calculated %d features; rolling warm-up removes %d rows", len(feature_names), rows_removed_for_features)

        horizon_periods = self._horizon_periods()
        targeted = self.target_builder.transform(featured, horizon_periods)
        usable = feature_ready & targeted["future_return"].notna() & targeted["target"].notna()
        final = targeted.loc[usable].reset_index(drop=True)
        rows_removed_for_target = int(feature_ready.sum()) - len(final)
        self.validator.validate_dataset(final, feature_names)
        distribution = self._target_distribution(final)
        LOGGER.info("Target distribution: %s", distribution)

        splits = self.splitter.split(final, horizon_periods)
        self.validator.validate_splits(splits)
        LOGGER.info(
            "Split sizes train=%d validation=%d test=%d; purged rows=%d",
            len(splits.train), len(splits.validation), len(splits.test), splits.purged_rows,
        )

        output = self.settings.ml_dataset.output_dir
        stem = f"{self.settings.market_data.symbol}_{self.settings.market_data.timeframe}"
        dataset_path = self.storage.save_frame(final, output / f"{stem}_ml_dataset.parquet")
        train_path = self.storage.save_frame(splits.train, output / f"{stem}_train.parquet")
        validation_path = self.storage.save_frame(splits.validation, output / f"{stem}_validation.parquet")
        test_path = self.storage.save_frame(splits.test, output / f"{stem}_test.parquet")
        manifest_path = self.storage.save_json(
            {"features": feature_names, "target": "target", "horizon_hours": self.settings.target.horizon_hours},
            output / "feature_manifest.json",
        )

        train_range = self._time_range(splits.train)
        validation_range = self._time_range(splits.validation)
        test_range = self._time_range(splits.test)
        split_metadata = lambda frame, time_range: {
            "rows": len(frame), "start": time_range[0].isoformat(), "end": time_range[1].isoformat()
        }
        metadata = {
            "symbol": self.settings.market_data.symbol,
            "timeframe": self.settings.market_data.timeframe,
            "rows_total": len(final),
            "features_count": len(feature_names),
            "target_distribution": distribution,
            "target": {
                "horizon_hours": self.settings.target.horizon_hours,
                "buy_threshold": self.settings.target.buy_threshold,
                "sell_threshold": self.settings.target.sell_threshold,
            },
            "purged_rows": splits.purged_rows,
            "train": split_metadata(splits.train, train_range),
            "validation": split_metadata(splits.validation, validation_range),
            "test": split_metadata(splits.test, test_range),
        }
        metadata_path = self.storage.save_json(metadata, output / "dataset_metadata.json")
        statistics_frame = final[feature_names].agg(["mean", "std", "min", "max"]).transpose()
        statistics = {
            feature: {stat: float(value) for stat, value in row.items()}
            for feature, row in statistics_frame.iterrows()
        }
        statistics_path = self.storage.save_json(statistics, output / "feature_statistics.json")
        LOGGER.info("ML dataset pipeline complete; dataset saved to %s", dataset_path)
        return MLDatasetReport(
            input_rows=input_rows,
            final_rows=len(final),
            feature_count=len(feature_names),
            rows_removed_for_features=rows_removed_for_features,
            rows_removed_for_target=rows_removed_for_target,
            target_distribution=distribution,
            train_rows=len(splits.train),
            validation_rows=len(splits.validation),
            test_rows=len(splits.test),
            train_range=train_range,
            validation_range=validation_range,
            test_range=test_range,
            purged_rows=splits.purged_rows,
            dataset_path=dataset_path,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            statistics_path=statistics_path,
        )
