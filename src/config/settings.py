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
class AppSettings:
    market_data: MarketDataSettings
    storage: StorageSettings
    http: HttpSettings
    logging: LoggingSettings


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
    )
