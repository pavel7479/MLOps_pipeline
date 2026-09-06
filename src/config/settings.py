"""Typed application configuration loaded from one YAML file."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class MarketDataSettings:
    provider: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime | None


@dataclass(frozen=True)
class StorageSettings:
    raw_dir: Path
    processed_dir: Path
    format: str


@dataclass(frozen=True)
class HttpSettings:
    timeout_seconds: float
    max_retries: int
    retry_delay_seconds: float


@dataclass(frozen=True)
class LoggingSettings:
    level: str


@dataclass(frozen=True)
class FeaturesSettings:
    return_periods: tuple[int, ...] = (1, 3, 6, 12, 24)
    rolling_mean_windows: tuple[int, ...] = (6, 12, 24)
    rolling_std_windows: tuple[int, ...] = (6, 12, 24)
    volume_change_periods: tuple[int, ...] = (1, 6)
    volume_rolling_windows: tuple[int, ...] = (6, 24)
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    atr_period: int = 14


@dataclass(frozen=True)
class TargetSettings:
    horizon_hours: int = 3
    buy_threshold: float = 0.003
    sell_threshold: float = -0.003
    rare_class_warning_threshold: float = 0.05


@dataclass(frozen=True)
class SplitSettings:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass(frozen=True)
class MLDatasetSettings:
    output_dir: Path = Path("data/ml")


@dataclass(frozen=True)
class AppSettings:
    market_data: MarketDataSettings
    storage: StorageSettings
    http: HttpSettings
    logging: LoggingSettings
    features: FeaturesSettings = FeaturesSettings()
    target: TargetSettings = TargetSettings()
    split: SplitSettings = SplitSettings()
    ml_dataset: MLDatasetSettings = MLDatasetSettings()


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ConfigurationError(f"Missing required setting: {section}.{key}")
    return mapping[key]


def _utc_datetime(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigurationError(f"Invalid datetime for {name}: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)



def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a list of positive integers") from exc
    if not result or any(item <= 0 for item in result):
        raise ConfigurationError(f"{name} must be a non-empty list of positive integers")
    return result

def load_settings(path: str | Path = "config.yaml") -> AppSettings:
    """Load, validate, and resolve application settings."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        market = raw["market_data"]
        storage = raw["storage"]
        http = raw["http"]
        logging_cfg = raw["logging"]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(f"Missing configuration section: {exc}") from exc
    base = config_path.parent
    start = _utc_datetime(_required(market, "start_date", "market_data"), "start_date")
    end_value = market.get("end_date")
    end = _utc_datetime(end_value, "end_date") if end_value is not None else None
    if end is not None and end <= start:
        raise ConfigurationError("market_data.end_date must be after start_date")
    fmt = str(_required(storage, "format", "storage")).lower()
    if fmt != "parquet":
        raise ConfigurationError("Only parquet storage is supported")
    retries = int(_required(http, "max_retries", "http"))
    if retries < 0:
        raise ConfigurationError("http.max_retries cannot be negative")
    feature_raw = raw.get("features", {})
    defaults = FeaturesSettings()
    returns_raw = feature_raw.get("returns", {})
    mean_raw = feature_raw.get("rolling_mean", {})
    std_raw = feature_raw.get("rolling_std", {})
    volume_raw = feature_raw.get("volume", {})
    rsi_raw = feature_raw.get("rsi", {})
    macd_raw = feature_raw.get("macd", {})
    atr_raw = feature_raw.get("atr", {})
    features = FeaturesSettings(
        return_periods=_positive_int_tuple(returns_raw.get("periods", defaults.return_periods), "features.returns.periods"),
        rolling_mean_windows=_positive_int_tuple(mean_raw.get("windows", defaults.rolling_mean_windows), "features.rolling_mean.windows"),
        rolling_std_windows=_positive_int_tuple(std_raw.get("windows", defaults.rolling_std_windows), "features.rolling_std.windows"),
        volume_change_periods=_positive_int_tuple(volume_raw.get("change_periods", defaults.volume_change_periods), "features.volume.change_periods"),
        volume_rolling_windows=_positive_int_tuple(volume_raw.get("rolling_windows", defaults.volume_rolling_windows), "features.volume.rolling_windows"),
        rsi_period=int(rsi_raw.get("period", defaults.rsi_period)),
        macd_fast_period=int(macd_raw.get("fast_period", defaults.macd_fast_period)),
        macd_slow_period=int(macd_raw.get("slow_period", defaults.macd_slow_period)),
        macd_signal_period=int(macd_raw.get("signal_period", defaults.macd_signal_period)),
        atr_period=int(atr_raw.get("period", defaults.atr_period)),
    )
    if min(features.rsi_period, features.macd_fast_period, features.macd_slow_period, features.macd_signal_period, features.atr_period) <= 0:
        raise ConfigurationError("Indicator periods must be positive")
    if features.macd_fast_period >= features.macd_slow_period:
        raise ConfigurationError("features.macd.fast_period must be below slow_period")
    target_raw = raw.get("target", {})
    target = TargetSettings(**{key: target_raw.get(key, getattr(TargetSettings(), key)) for key in TargetSettings.__dataclass_fields__})
    if target.horizon_hours <= 0 or target.sell_threshold >= target.buy_threshold:
        raise ConfigurationError("Target horizon must be positive and sell_threshold below buy_threshold")
    split_raw = raw.get("split", {})
    split = SplitSettings(**{key: float(split_raw.get(key, getattr(SplitSettings(), key))) for key in SplitSettings.__dataclass_fields__})
    if any(value <= 0 for value in (split.train_ratio, split.validation_ratio, split.test_ratio)) or abs(sum((split.train_ratio, split.validation_ratio, split.test_ratio)) - 1.0) > 1e-9:
        raise ConfigurationError("Split ratios must be positive and sum to 1.0")
    output_dir = (base / raw.get("ml_dataset", {}).get("output_dir", "data/ml")).resolve()
    return AppSettings(
        market_data=MarketDataSettings(
            provider=str(_required(market, "provider", "market_data")).lower(),
            symbol=str(_required(market, "symbol", "market_data")).upper(),
            timeframe=str(_required(market, "timeframe", "market_data")),
            start_date=start,
            end_date=end,
        ),
        storage=StorageSettings(
            raw_dir=(base / _required(storage, "raw_dir", "storage")).resolve(),
            processed_dir=(base / _required(storage, "processed_dir", "storage")).resolve(),
            format=fmt,
        ),
        http=HttpSettings(
            timeout_seconds=float(_required(http, "timeout_seconds", "http")),
            max_retries=retries,
            retry_delay_seconds=float(_required(http, "retry_delay_seconds", "http")),
        ),
        logging=LoggingSettings(level=str(_required(logging_cfg, "level", "logging")).upper()),
        features=features,
        target=target,
        split=split,
        ml_dataset=MLDatasetSettings(output_dir),
    )
