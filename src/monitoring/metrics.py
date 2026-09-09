"""Explicit Prometheus collectors with injectable registries."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)
INFERENCE_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)
PREDICTIONS = ("BUY", "HOLD", "SELL")


class ApplicationMetrics:
    """Metrics owned by one FastAPI application instance."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.http_requests = Counter(
            "crypto_http_requests_total",
            "HTTP requests completed by the inference API.",
            ("method", "route", "status_code"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "crypto_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("method", "route"),
            buckets=HTTP_BUCKETS,
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "crypto_ml_inference_duration_seconds",
            "Time spent only inside model.predict.",
            ("model_version",),
            buckets=INFERENCE_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "crypto_ml_predictions_total",
            "New prediction rows committed successfully.",
            ("prediction", "model_version"),
            registry=self.registry,
        )
        self.replays = Counter(
            "crypto_ml_prediction_replays_total",
            "Idempotent prediction request replays.",
            registry=self.registry,
        )
        self.conflicts = Counter(
            "crypto_ml_prediction_conflicts_total",
            "request_id conflicts with a different payload.",
            registry=self.registry,
        )
        self.database_write_failures = Counter(
            "crypto_database_write_failures_total",
            "Failed attempts to commit prediction rows.",
            registry=self.registry,
        )
        self.loaded_model_info = Gauge(
            "crypto_loaded_model_info",
            "Identity and backtest gate of the model loaded by the API.",
            ("registered_model", "alias", "version", "backtest_status"),
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "crypto_model_loaded",
            "Whether the API has loaded a usable model.",
            registry=self.registry,
        )
        self.model_loaded.set(0)

    def set_loaded_model(
        self,
        registered_model: str,
        alias: str,
        version: str,
        backtest_status: str | None,
    ) -> None:
        self.loaded_model_info.clear()
        self.loaded_model_info.labels(
            registered_model=registered_model,
            alias=alias,
            version=version,
            backtest_status=backtest_status or "unknown",
        ).set(1)
        self.model_loaded.set(1)

    def observe_inference(self, version: str, seconds: float) -> None:
        self.inference_duration.labels(model_version=version).observe(seconds)

    def record_prediction(self, prediction: str, version: str) -> None:
        self.predictions.labels(
            prediction=prediction, model_version=version
        ).inc()

    def record_replay(self) -> None:
        self.replays.inc()

    def record_conflict(self) -> None:
        self.conflicts.inc()

    def record_database_write_failure(self) -> None:
        self.database_write_failures.inc()


class WorkerMetrics:
    """Metrics owned by one monitoring worker process."""

    def __init__(
        self, feature_names: tuple[str, ...], registry: CollectorRegistry | None = None
    ) -> None:
        self.registry = registry or CollectorRegistry()
        self.recent_prediction_share = Gauge(
            "crypto_recent_prediction_share",
            "Share of each class in the bounded recent prediction window.",
            ("prediction",),
            registry=self.registry,
        )
        self.window_samples = Gauge(
            "crypto_monitoring_window_samples",
            "Prediction rows currently included in monitoring.",
            registry=self.registry,
        )
        self.drift_available = Gauge(
            "crypto_feature_drift_available",
            "Whether enough recent rows exist to calculate PSI.",
            registry=self.registry,
        )
        self.feature_drift_score = Gauge(
            "crypto_feature_drift_score",
            "Population Stability Index for each feature.",
            ("feature",),
            registry=self.registry,
        )
        self.drift_warning_features = Gauge(
            "crypto_drift_warning_features",
            "Number of features in warning drift state.",
            registry=self.registry,
        )
        self.drift_critical_features = Gauge(
            "crypto_drift_critical_features",
            "Number of features in critical drift state.",
            registry=self.registry,
        )
        self.last_drift_check = Gauge(
            "crypto_last_drift_check_timestamp_seconds",
            "Unix timestamp of the last successful monitoring cycle.",
            registry=self.registry,
        )
        self.database_reachable = Gauge(
            "crypto_database_reachable",
            "Whether PostgreSQL was reachable in the latest worker cycle.",
            registry=self.registry,
        )
        self.mlflow_reachable = Gauge(
            "crypto_mlflow_reachable",
            "Whether MLflow was reachable in the latest worker cycle.",
            registry=self.registry,
        )
        for prediction in PREDICTIONS:
            self.recent_prediction_share.labels(prediction=prediction).set(0)
        for feature in feature_names:
            self.feature_drift_score.labels(feature=feature).set(0)
        self.window_samples.set(0)
        self.drift_available.set(0)
        self.drift_warning_features.set(0)
        self.drift_critical_features.set(0)
        self.database_reachable.set(0)
        self.mlflow_reachable.set(0)
