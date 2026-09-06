"""Chronological train/validation/test splitting with boundary purge."""

import pandas as pd

from src.config.settings import SplitSettings

from .models import DatasetSplits


class ChronologicalDatasetSplitter:
    """Split ordered rows without shuffling and purge target-boundary leakage."""

    def __init__(self, settings: SplitSettings) -> None:
        self.settings = settings

    def split(self, data: pd.DataFrame, purge_periods: int) -> DatasetSplits:
        """Split by position and remove boundary rows from train and validation."""
        size = len(data)
        train_boundary = int(size * self.settings.train_ratio)
        validation_boundary = int(size * (self.settings.train_ratio + self.settings.validation_ratio))
        train_stop = train_boundary - purge_periods
        validation_stop = validation_boundary - purge_periods
        if train_stop <= 0 or validation_stop <= train_boundary or validation_boundary >= size:
            raise ValueError("Dataset is too small for configured split ratios and purge horizon")
        train = data.iloc[:train_stop].copy().reset_index(drop=True)
        validation = data.iloc[train_boundary:validation_stop].copy().reset_index(drop=True)
        test = data.iloc[validation_boundary:].copy().reset_index(drop=True)
        return DatasetSplits(train, validation, test, purge_periods * 2)
