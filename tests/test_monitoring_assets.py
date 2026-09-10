import json
from pathlib import Path

import yaml

from src.config import load_settings
from src.features import FeatureEngineer
from src.monitoring.reference import load_reference


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "monitoring/grafana/dashboards"


def _load_dashboards() -> dict[str, dict]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(DASHBOARD_DIR.glob("*.json"))
    }


def _panel(dashboard: dict, title: str) -> dict:
    return next(panel for panel in dashboard["panels"] if panel["title"] == title)


def _value_mapping(panel: dict) -> dict:
    mappings = panel["fieldConfig"]["defaults"]["mappings"]
    return next(
        mapping["options"] for mapping in mappings if mapping["type"] == "value"
    )


def _organize_exclusions(panel: dict) -> set[str]:
    organize = next(
        transformation
        for transformation in panel["transformations"]
        if transformation["id"] == "organize"
    )
    return {
        name
        for name, excluded in organize["options"]["excludeByName"].items()
        if excluded
    }


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
        "CryptoApiDown",
        "MonitoringWorkerDown",
        "DatabaseUnavailable",
        "MlflowUnavailable",
        "HighHttpErrorRate",
        "HighApiLatency",
        "CriticalFeatureDrift",
        "PredictionClassDominance",
    }


def test_grafana_provisioning_and_dashboards_are_valid_and_unique() -> None:
    datasource = yaml.safe_load(
        (
            ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml"
        ).read_text(encoding="utf-8")
    )
    assert datasource["datasources"][0]["url"] == "http://prometheus:9090"
    assert datasource["datasources"][0]["uid"] == "crypto-prometheus"
    dashboards = _load_dashboards()
    assert {dashboard["title"] for dashboard in dashboards.values()} == {
        "Crypto ML — Service Overview",
        "Crypto ML — Model Monitoring",
    }
    assert {dashboard["uid"] for dashboard in dashboards.values()} == {
        "crypto-ml-service-overview",
        "crypto-ml-model-monitoring",
    }
    titles = {
        panel["title"]
        for dashboard in dashboards.values()
        for panel in dashboard["panels"]
    }
    assert {
        "API STATUS",
        "HTTP LATENCY",
        "ML INFERENCE P95",
        "CURRENT PREDICTION DISTRIBUTION",
        "FEATURE PSI TABLE",
        "TOP 5 DRIFTING FEATURES",
    } <= titles

    for dashboard in dashboards.values():
        panel_ids = [panel["id"] for panel in dashboard["panels"]]
        assert len(panel_ids) == len(set(panel_ids))
        assert dashboard["editable"] is False
        assert dashboard["refresh"] == "15s"
        assert dashboard["timezone"] == "browser"
        for index, panel in enumerate(dashboard["panels"]):
            left = panel["gridPos"]
            for other in dashboard["panels"][index + 1:]:
                right = other["gridPos"]
                overlaps = (
                    left["x"] < right["x"] + right["w"]
                    and right["x"] < left["x"] + left["w"]
                    and left["y"] < right["y"] + right["h"]
                    and right["y"] < left["y"] + left["h"]
                )
                assert not overlaps, (
                    f"{dashboard['uid']}: {panel['title']} overlaps "
                    f"{other['title']}"
                )
        for panel in dashboard["panels"]:
            grid = panel["gridPos"]
            assert grid["h"] > 0 and grid["w"] > 0
            assert 0 <= grid["x"] < 24
            assert grid["x"] + grid["w"] <= 24
            assert grid["y"] >= 0
            if panel["type"] == "row":
                continue
            assert panel["datasource"]["uid"] == "crypto-prometheus"
            assert panel.get("description")
            for target in panel["targets"]:
                assert target["datasource"]["uid"] == "crypto-prometheus"


def test_grafana_status_cards_have_human_readable_mappings() -> None:
    dashboards = _load_dashboards()
    service = dashboards["service_overview.json"]
    model = dashboards["ml_monitoring.json"]

    expected = {
        "API STATUS": ("DOWN", "UP"),
        "WORKER STATUS": ("DOWN", "UP"),
        "POSTGRESQL": ("UNAVAILABLE", "AVAILABLE"),
        "MLFLOW": ("UNAVAILABLE", "AVAILABLE"),
    }
    for title, (zero, one) in expected.items():
        mapping = _value_mapping(_panel(service, title))
        assert mapping["0"]["text"] == zero
        assert mapping["1"]["text"] == one

    drift_mapping = _value_mapping(_panel(model, "DRIFT STATUS"))
    assert drift_mapping["0"]["text"] == "INSUFFICIENT DATA"
    assert drift_mapping["0"]["color"] == "gray"
    assert drift_mapping["1"]["text"] == "READY"
    assert drift_mapping["1"]["color"] == "green"


def test_grafana_tables_hide_prometheus_technical_fields() -> None:
    dashboards = _load_dashboards()
    service = dashboards["service_overview.json"]
    model = dashboards["ml_monitoring.json"]
    required_hidden = {"Time", "__name__", "instance", "job"}

    assert required_hidden <= _organize_exclusions(_panel(service, "CURRENT MODEL"))
    assert required_hidden <= _organize_exclusions(_panel(model, "LOADED MODEL"))
    psi_table = _panel(model, "FEATURE PSI TABLE")
    assert required_hidden <= _organize_exclusions(psi_table)

    sort = next(
        item for item in psi_table["transformations"] if item["id"] == "sortBy"
    )
    assert sort["options"]["sort"] == [{"desc": True, "field": "PSI"}]


def test_grafana_drift_panels_distinguish_insufficient_data() -> None:
    model = _load_dashboards()["ml_monitoring.json"]
    for title in {
        "WARNING FEATURES",
        "CRITICAL FEATURES",
        "TOP 5 DRIFTING FEATURES",
        "FEATURE PSI TABLE",
        "FEATURE PSI OVER TIME",
    }:
        expression = _panel(model, title)["targets"][0]["expr"]
        assert "crypto_feature_drift_available == 1" in expression

    last_check = _panel(model, "LAST DRIFT CHECK")
    expression = last_check["targets"][0]["expr"]
    assert "crypto_last_drift_check_timestamp_seconds" in expression
    assert "* 1000" in expression
    assert last_check["fieldConfig"]["defaults"]["unit"] == "dateTimeAsIso"
    assert last_check["fieldConfig"]["defaults"]["noValue"] == "NEVER"


def test_grafana_prediction_display_and_feature_selector() -> None:
    dashboards = _load_dashboards()
    service_predictions = _panel(
        dashboards["service_overview.json"], "COMMITTED PREDICTIONS"
    )
    model_predictions = _panel(
        dashboards["ml_monitoring.json"], "CURRENT PREDICTION DISTRIBUTION"
    )
    for panel in (service_predictions, model_predictions):
        assert panel["type"] == "bargauge"
        expression = panel["targets"][0]["expr"]
        assert all(label in expression for label in ('"BUY"', '"HOLD"', '"SELL"'))

    model = dashboards["ml_monitoring.json"]
    variables = model["templating"]["list"]
    assert len(variables) == 1
    feature = variables[0]
    assert feature["name"] == "Feature"
    assert feature["includeAll"] is True
    assert feature["allValue"] == ".*"
    assert feature["query"]["query"] == (
        "label_values(crypto_feature_drift_score, feature)"
    )
    history_query = _panel(model, "FEATURE PSI OVER TIME")["targets"][0]["expr"]
    assert "$Feature" in history_query


def test_grafana_dashboards_have_required_operational_sections() -> None:
    dashboards = _load_dashboards()
    service_rows = {
        panel["title"]
        for panel in dashboards["service_overview.json"]["panels"]
        if panel["type"] == "row"
    }
    model_rows = {
        panel["title"]
        for panel in dashboards["ml_monitoring.json"]["panels"]
        if panel["type"] == "row"
    }
    assert service_rows == {
        "SYSTEM STATUS",
        "TRAFFIC & ERRORS",
        "LATENCY",
        "PREDICTIONS",
    }
    assert model_rows == {
        "MODEL & DATA STATUS",
        "RECENT PREDICTIONS",
        "FEATURE DRIFT",
        "DRIFT HISTORY",
    }


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
