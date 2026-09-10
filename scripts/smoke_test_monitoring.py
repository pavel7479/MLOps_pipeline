"""Verify the containerized API -> worker -> Prometheus -> Grafana chain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from uuid import uuid4

import httpx


API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_BASE_URL", "http://127.0.0.1:9090"
).rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_BASE_URL", "http://127.0.0.1:3000").rstrip("/")
SAMPLE_PATH = Path(os.getenv(
    "INFERENCE_SAMPLE_PATH", "tests/fixtures/inference_sample.json"
))


def _prometheus_query(client: httpx.Client, query: str) -> list[dict]:
    response = client.get(
        f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "success", payload
    return payload["data"]["result"]


def _grafana_dashboard(
    client: httpx.Client,
    uid: str,
    auth: tuple[str, str],
) -> dict:
    response = client.get(f"{GRAFANA_URL}/api/dashboards/uid/{uid}", auth=auth)
    response.raise_for_status()
    payload = response.json()
    assert payload["meta"]["provisioned"] is True, payload["meta"]
    return payload["dashboard"]


def _wait_for_metrics(client: httpx.Client, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    last_samples = 0.0
    while time.monotonic() < deadline:
        samples = _prometheus_query(client, "crypto_monitoring_window_samples")
        if samples:
            last_samples = float(samples[0]["value"][1])
        predictions = _prometheus_query(
            client, "sum(crypto_ml_predictions_total)"
        )
        if last_samples >= 5 and predictions and float(predictions[0]["value"][1]) >= 5:
            return
        time.sleep(2)
    raise TimeoutError(
        f"Monitoring metrics did not converge; last window samples={last_samples}"
    )


def _wait_for_targets(client: httpx.Client, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    states = {}
    while time.monotonic() < deadline:
        response = client.get(f"{PROMETHEUS_URL}/api/v1/targets")
        response.raise_for_status()
        active = response.json()["data"]["activeTargets"]
        states = {
            target["labels"]["job"]: target["health"] for target in active
        }
        if states.get("crypto-api") == "up" and states.get(
            "crypto-ml-monitor"
        ) == "up":
            return
        time.sleep(2)
    raise TimeoutError(f"Prometheus targets did not become UP: {states}")


def main() -> int:
    with httpx.Client(timeout=10) as client:
        api_metrics = client.get(f"{API_URL}/metrics")
        api_metrics.raise_for_status()
        assert "crypto_http_requests_total" in api_metrics.text
        assert "crypto_model_loaded 1.0" in api_metrics.text

        health = client.get(f"{PROMETHEUS_URL}/-/healthy")
        health.raise_for_status()
        _wait_for_targets(client)

        sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        for _ in range(6):
            payload = {**sample, "request_id": str(uuid4())}
            response = client.post(f"{API_URL}/api/v1/predict", json=payload)
            response.raise_for_status()
            assert response.json()["replayed"] is False
        _wait_for_metrics(client)
        availability_result = _prometheus_query(
            client, "crypto_feature_drift_available"
        )
        assert availability_result
        drift_available = float(availability_result[0]["value"][1])
        gated_psi = _prometheus_query(
            client,
            "crypto_feature_drift_score "
            "and on() (crypto_feature_drift_available == 1)",
        )
        assert bool(gated_psi) is bool(drift_available)

        timestamp = _prometheus_query(
            client,
            "(crypto_last_drift_check_timestamp_seconds > 0) * 1000",
        )
        assert timestamp and float(timestamp[0]["value"][1]) > 1_000_000_000_000

        distribution = _prometheus_query(
            client,
            "max by (prediction) (crypto_recent_prediction_share) "
            'or label_replace(vector(0), "prediction", "BUY", "", ".*") '
            'or label_replace(vector(0), "prediction", "HOLD", "", ".*") '
            'or label_replace(vector(0), "prediction", "SELL", "", ".*")',
        )
        assert {item["metric"]["prediction"] for item in distribution} == {
            "BUY",
            "HOLD",
            "SELL",
        }

        rules = client.get(f"{PROMETHEUS_URL}/api/v1/rules", params={"type": "alert"})
        rules.raise_for_status()
        names = {
            rule["name"]
            for group in rules.json()["data"]["groups"]
            for rule in group["rules"]
        }
        required = {
            "CryptoApiDown", "MonitoringWorkerDown", "DatabaseUnavailable",
            "MlflowUnavailable", "HighHttpErrorRate", "HighApiLatency",
            "CriticalFeatureDrift", "PredictionClassDominance",
        }
        assert required <= names, names

        grafana_health = client.get(f"{GRAFANA_URL}/api/health")
        grafana_health.raise_for_status()
        auth = (
            os.getenv("GRAFANA_ADMIN_USER", "admin"),
            os.environ["GRAFANA_ADMIN_PASSWORD"],
        )
        datasource = client.get(
            f"{GRAFANA_URL}/api/datasources/uid/crypto-prometheus", auth=auth
        )
        datasource.raise_for_status()
        assert datasource.json()["url"] == "http://prometheus:9090"
        dashboards = client.get(
            f"{GRAFANA_URL}/api/search", params={"type": "dash-db"}, auth=auth
        )
        dashboards.raise_for_status()
        uids = {item["uid"] for item in dashboards.json()}
        assert {
            "crypto-ml-service-overview", "crypto-ml-model-monitoring"
        } <= uids
        service_dashboard = _grafana_dashboard(
            client, "crypto-ml-service-overview", auth
        )
        model_dashboard = _grafana_dashboard(
            client, "crypto-ml-model-monitoring", auth
        )
        assert service_dashboard["title"] == "Crypto ML — Service Overview"
        assert model_dashboard["title"] == "Crypto ML — Model Monitoring"
        service_panels = {
            panel["title"]: panel for panel in service_dashboard["panels"]
        }
        model_panels = {
            panel["title"]: panel for panel in model_dashboard["panels"]
        }
        assert {
            "SYSTEM STATUS",
            "TRAFFIC & ERRORS",
            "LATENCY",
            "PREDICTIONS",
            "API STATUS",
            "CURRENT MODEL",
        } <= set(service_panels)
        assert {
            "MODEL & DATA STATUS",
            "RECENT PREDICTIONS",
            "FEATURE DRIFT",
            "DRIFT HISTORY",
            "DRIFT STATUS",
            "FEATURE PSI TABLE",
        } <= set(model_panels)
        last_check_query = model_panels["LAST DRIFT CHECK"]["targets"][0]["expr"]
        assert "* 1000" in last_check_query
        psi_query = model_panels["FEATURE PSI TABLE"]["targets"][0]["expr"]
        assert "crypto_feature_drift_available == 1" in psi_query
    print("Monitoring smoke test passed: API, worker, Prometheus, alerts, Grafana.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
