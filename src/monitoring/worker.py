"""Resilient standalone monitoring worker with a Prometheus endpoint."""

from __future__ import annotations

from collections import Counter
import logging
import time
from typing import Callable
from urllib.request import urlopen

from prometheus_client import start_http_server

from src.config import AppSettings, load_settings
from src.database import create_database
from src.features import FeatureEngineer

from .drift import calculate_drift
from .metrics import PREDICTIONS, WorkerMetrics
from .reference import MonitoringReference, load_reference
from .repository import MonitoringRepository

LOGGER = logging.getLogger(__name__)


def mlflow_is_reachable(uri: str) -> bool:
    try:
        with urlopen(f"{uri.rstrip('/')}/health", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


class MonitoringWorker:
    """Refresh gauges without terminating on external service failures."""

    def __init__(
        self,
        repository: MonitoringRepository,
        reference: MonitoringReference,
        metrics: WorkerMetrics,
        settings: AppSettings,
        *,
        mlflow_check: Callable[[str], bool] = mlflow_is_reachable,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.repository = repository
        self.reference = reference
        self.metrics = metrics
        self.settings = settings
        self.mlflow_check = mlflow_check
        self.clock = clock

    def run_cycle(self) -> bool:
        mlflow_reachable = self.mlflow_check(self.settings.mlflow.tracking_uri)
        self.metrics.mlflow_reachable.set(int(mlflow_reachable))
        if not mlflow_reachable:
            LOGGER.warning("MLflow is unavailable; drift monitoring continues")
        try:
            rows = self.repository.fetch_recent(self.settings.monitoring.window_size)
        except Exception:
            self.metrics.database_reachable.set(0)
            LOGGER.exception("PostgreSQL monitoring query failed; retrying next cycle")
            return False

        self.metrics.database_reachable.set(1)
        self.metrics.window_samples.set(len(rows))
        counts = Counter(row.prediction for row in rows)
        denominator = len(rows)
        for prediction in PREDICTIONS:
            share = counts[prediction] / denominator if denominator else 0
            self.metrics.recent_prediction_share.labels(
                prediction=prediction
            ).set(share)

        result = calculate_drift(
            [row.features for row in rows],
            self.reference,
            min_samples=self.settings.monitoring.min_samples,
            warning_threshold=self.settings.monitoring.warning_threshold,
            critical_threshold=self.settings.monitoring.critical_threshold,
        )
        self.metrics.drift_available.set(int(result.available))
        self.metrics.drift_warning_features.set(result.warning_features)
        self.metrics.drift_critical_features.set(result.critical_features)
        for feature in self.reference.feature_names:
            self.metrics.feature_drift_score.labels(feature=feature).set(
                result.scores.get(feature, 0)
            )
        self.metrics.last_drift_check.set(self.clock())
        LOGGER.info(
            "Monitoring cycle samples=%d drift=%s warning=%d critical=%d",
            len(rows), result.status, result.warning_features, result.critical_features,
        )
        return True


def create_worker(settings: AppSettings) -> tuple[MonitoringWorker, object]:
    expected_features = tuple(FeatureEngineer(settings.features).feature_names)
    reference = load_reference(
        settings.monitoring.reference_path, expected_features=expected_features
    )
    database = create_database(settings.database)
    metrics = WorkerMetrics(reference.feature_names)
    worker = MonitoringWorker(
        MonitoringRepository(database.session_factory), reference, metrics, settings
    )
    return worker, database


def main() -> int:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        worker, database = create_worker(settings)
    except Exception:
        LOGGER.exception("Monitoring worker startup validation failed")
        return 1
    start_http_server(
        settings.monitoring.metrics_port,
        addr=settings.monitoring.metrics_host,
        registry=worker.metrics.registry,
    )
    LOGGER.info(
        "Monitoring metrics listening on %s:%d",
        settings.monitoring.metrics_host,
        settings.monitoring.metrics_port,
    )
    try:
        while True:
            worker.run_cycle()
            time.sleep(settings.monitoring.poll_interval_seconds)
    except KeyboardInterrupt:
        LOGGER.info("Monitoring worker stopped")
    finally:
        database.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
