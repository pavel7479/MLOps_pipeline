import json
from pathlib import Path

import yaml

from src.features import FeatureEngineer
from src.config import load_settings
from src.monitoring.reference import load_reference


ROOT = Path(__file__).resolve().parents[1]


def test_versioned_reference_matches_exact_feature_contract() -> None:
    settings = load_settings(ROOT / "config.yaml")
    expected = tuple(FeatureEngineer(settings.features).feature_names)
    reference = load_reference(
        ROOT / "monitoring/reference/BTCUSDT_1h_train_reference.json", expected
    )
    assert len(reference.feature_names) == 28
    assert reference.metadata["symbol"] == "BTCUSDT"
    assert reference.metadata["timeframe"] == "1h"
    assert reference.metadata["train_rows"] > 0
    assert len(reference.metadata["train_sha256"]) == 64
    assert len(reference.metadata["manifest_sha256"]) == 64


def test_prometheus_scrapes_only_api_and_worker_and_has_required_alerts() -> None:
    config = yaml.safe_load(
        (ROOT / "monitoring/prometheus/prometheus.yml").read_text(encoding="utf-8")
    )
    jobs = {item["job_name"]: item for item in config["scrape_configs"]}
    assert set(jobs) == {"crypto-api", "crypto-ml-monitor"}
    assert jobs["crypto-api"]["static_configs"][0]["targets"] == ["api:8000"]
    assert jobs["crypto-ml-monitor"]["static_configs"][0]["targets"] == [
        "monitoring-worker:9101"
    ]
    rules = yaml.safe_load(
        (ROOT / "monitoring/prometheus/rules.yml").read_text(encoding="utf-8")
    )
    names = {rule["alert"] for group in rules["groups"] for rule in group["rules"]}
    assert names == {
        "CryptoApiDown", "MonitoringWorkerDown", "DatabaseUnavailable",
        "MlflowUnavailable", "HighHttpErrorRate", "HighApiLatency",
        "CriticalFeatureDrift", "PredictionClassDominance",
    }


def test_grafana_provisioning_and_dashboards_are_valid_and_unique() -> None:
    datasource = yaml.safe_load(
        (ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml").read_text()
    )
    assert datasource["datasources"][0]["url"] == "http://prometheus:9090"
    assert datasource["datasources"][0]["uid"] == "crypto-prometheus"
    dashboards = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "monitoring/grafana/dashboards").glob("*.json"))
    ]
    assert {dashboard["title"] for dashboard in dashboards} == {
        "Crypto ML — Service Overview", "Crypto ML — Model Monitoring"
    }
    assert len({dashboard["uid"] for dashboard in dashboards}) == 2
    titles = {panel["title"] for dashboard in dashboards for panel in dashboard["panels"]}
    assert {
        "API status", "HTTP latency p50 / p95 / p99", "ML inference p95",
        "Recent prediction shares", "Feature PSI — all 28", "Top 5 feature PSI",
    } <= titles


def test_compose_isolated_monitoring_topology_and_persistence() -> None:
    compose = yaml.load((ROOT / "compose.yaml").read_text(), Loader=yaml.BaseLoader)
    services = compose["services"]
    assert {"monitoring-worker", "prometheus", "grafana"} <= set(services)
    assert services["prometheus"]["image"] == "prom/prometheus:v3.14.0"
    assert services["grafana"]["image"] == "grafana/grafana:13.2.1"
    assert "ports" not in services["monitoring-worker"]
    assert "monitoring-worker" not in services["api"].get("depends_on", {})
    assert "prometheus" not in services["api"].get("depends_on", {})
    assert "grafana" not in services["api"].get("depends_on", {})
    assert {"postgres_data", "prometheus_data", "grafana_data"} <= set(
        compose["volumes"]
    )
    assert services["monitoring-worker"]["image"] == services["api"]["image"]


def test_runtime_copies_reference_but_not_ml_datasets() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=appuser:appuser monitoring/reference" in dockerfile
    assert "data/ml/BTCUSDT_1h_train.parquet" not in dockerfile
    assert "data/ml/BTCUSDT_1h_validation.parquet" not in dockerfile
    assert "data/ml/BTCUSDT_1h_test.parquet" not in dockerfile
