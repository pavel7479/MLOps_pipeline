"""Configuration-driven orchestration for multiple market datasets."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Callable, Iterable

from src.config.settings import AppSettings

from .pipeline import MarketDataPipeline
from .provider import MarketDataProvider
from .storage import ParquetStorage, sha256_file
from .validator import MarketDataValidator

LOGGER = logging.getLogger(__name__)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class DatasetFailure:
    symbol: str
    interval: str
    error: str


@dataclass(frozen=True)
class DatasetSummary:
    symbol: str
    interval: str
    row_count: int
    start_timestamp: datetime
    end_timestamp: datetime
    duplicates: int
    missing_intervals: int
    validation_status: str
    file_path: Path
    file_hash: str


@dataclass(frozen=True)
class MultiAssetIngestionResult:
    manifest_path: Path
    successful_datasets: int
    requested_datasets: int
    datasets: tuple[DatasetSummary, ...]
    failures: tuple[DatasetFailure, ...]

    @property
    def is_success(self) -> bool:
        return not self.failures and self.successful_datasets == self.requested_datasets


class MultiAssetMarketDataPipeline:
    """Run each configured symbol/interval pair and maintain an atomic manifest."""

    MANIFEST_NAME = "market_data_manifest.json"

    def __init__(
        self,
        settings: AppSettings,
        provider: MarketDataProvider | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.pipeline = MarketDataPipeline(settings, provider=provider, now=self.now)
        self.storage = ParquetStorage()
        self.validator = MarketDataValidator()

    def run(
        self,
        symbols: Iterable[str] | None = None,
        intervals: Iterable[str] | None = None,
    ) -> MultiAssetIngestionResult:
        selected_symbols = tuple(symbols or self.settings.market_data.configured_symbols)
        selected_intervals = tuple(intervals or self.settings.market_data.configured_intervals)
        requested = tuple(
            (symbol.upper(), interval)
            for symbol in selected_symbols
            for interval in selected_intervals
        )
        if not requested:
            raise ValueError("At least one symbol/interval pair is required")

        manifest_path = self.settings.storage.processed_dir / self.MANIFEST_NAME
        entries = self._existing_entries(manifest_path)
        failures: list[DatasetFailure] = []
        summaries: list[DatasetSummary] = []
        successful = 0
        run_created_at = _utc_iso(self.now())

        for symbol, interval in requested:
            LOGGER.info("Dataset %s %s: STARTED", symbol, interval)
            try:
                report = self.pipeline.run_dataset(symbol, interval)
                final = self.storage.load(report.processed_path)
                validation = self.validator.validate(final, interval)
                fatal_errors = [
                    error
                    for error in validation.errors
                    if not error.startswith("Missing intervals:")
                ]
                if fatal_errors:
                    raise ValueError("; ".join(fatal_errors))
                file_hash = sha256_file(report.processed_path)
                summaries.append(DatasetSummary(
                    symbol=symbol,
                    interval=interval,
                    row_count=len(final),
                    start_timestamp=report.start_timestamp,
                    end_timestamp=report.end_timestamp,
                    duplicates=validation.duplicate_timestamps,
                    missing_intervals=validation.missing_intervals,
                    validation_status="PASS",
                    file_path=report.processed_path,
                    file_hash=file_hash,
                ))
                entries[(symbol, interval)] = {
                    "symbol": symbol,
                    "interval": interval,
                    "row_count": len(final),
                    "first_timestamp": _utc_iso(report.start_timestamp),
                    "last_timestamp": _utc_iso(report.end_timestamp),
                    "requested_start": _utc_iso(self.settings.market_data.start_date),
                    "actual_start": _utc_iso(report.start_timestamp),
                    "duplicate_timestamps": validation.duplicate_timestamps,
                    "missing_intervals": validation.missing_intervals,
                    "validation_status": "PASS",
                    "file_path": str(report.processed_path),
                    "file_hash": file_hash,
                    "created_at": run_created_at,
                    "provider": self.settings.market_data.provider,
                }
                successful += 1
                LOGGER.info(
                    "Dataset %s %s: PASS rows=%d gaps=%d path=%s",
                    symbol,
                    interval,
                    len(final),
                    validation.missing_intervals,
                    report.processed_path,
                )
            except Exception as exc:
                failure = DatasetFailure(symbol, interval, str(exc))
                failures.append(failure)
                LOGGER.exception("Dataset %s %s: FAILED: %s", symbol, interval, exc)

        payload = {
            "schema_version": 1,
            "provider": self.settings.market_data.provider,
            "generated_at": run_created_at,
            "datasets": [entries[key] for key in sorted(entries)],
            "last_run": {
                "requested_datasets": len(requested),
                "successful_datasets": successful,
                "failed_datasets": len(failures),
                "status": "PASS" if not failures else "FAIL",
                "failures": [asdict(failure) for failure in failures],
            },
        }
        self._write_manifest(manifest_path, payload)
        return MultiAssetIngestionResult(
            manifest_path=manifest_path,
            successful_datasets=successful,
            requested_datasets=len(requested),
            datasets=tuple(summaries),
            failures=tuple(failures),
        )

    @staticmethod
    def _existing_entries(path: Path) -> dict[tuple[str, str], dict[str, object]]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                (str(item["symbol"]), str(item["interval"])): item
                for item in payload.get("datasets", [])
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Cannot read existing manifest {path}: {exc}") from exc

    @staticmethod
    def _write_manifest(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
