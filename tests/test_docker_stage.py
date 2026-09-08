from pathlib import Path
from types import SimpleNamespace

import yaml

from src.config import load_settings
from src.database import get_database_url
from src.mlops import BootstrapSource, ensure_registry_aliases

ROOT = Path(__file__).resolve().parents[1]


class FakeRegistryGateway:
    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = dict(aliases or {})
        self.assignments: list[tuple[str, str]] = []

    def get_alias(self, alias: str):
        version = self.aliases.get(alias)
        return None if version is None else SimpleNamespace(version=version)

    def set_alias(self, alias: str, version: str) -> None:
        self.aliases[alias] = version
        self.assignments.append((alias, version))


def _sources(tmp_path: Path) -> tuple[BootstrapSource, ...]:
    return (
        BootstrapSource(
            "champion",
            "baseline",
            tmp_path / "lightgbm.joblib",
            "block_3",
            "baseline-key",
        ),
        BootstrapSource(
            "challenger",
            "tuned",
            tmp_path / "lightgbm_tuned.joblib",
            "block_4",
            "tuned-key",
        ),
    )


def test_docker_environment_overrides_internal_service_hosts(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://mlops_user:secret@postgres:5432/mlops_pipeline",
    )
    settings = load_settings(ROOT / "config.yaml")
    assert settings.mlflow.tracking_uri == "http://mlflow:5000"
    assert "@postgres:5432/" in get_database_url(settings.database)


def test_compose_uses_pinned_internal_services_and_ordering() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert {"postgres", "mlflow", "migrate", "mlflow-init", "api"} <= set(services)
    assert services["postgres"]["image"] == "postgres:17.11"
    assert services["mlflow"]["image"] == "ghcr.io/mlflow/mlflow:v3.16.0"
    assert "ports" not in services["postgres"]
    assert "@postgres:5432/" in services["api"]["environment"]["DATABASE_URL"]
    assert (
        services["api"]["environment"]["MLFLOW_TRACKING_URI"]
        == "http://mlflow:5000"
    )
    assert (
        services["api"]["depends_on"]["migrate"]["condition"]
        == "service_completed_successfully"
    )
    assert (
        services["api"]["depends_on"]["mlflow-init"]["condition"]
        == "service_completed_successfully"
    )
    assert services["api"]["healthcheck"]["test"][-1].endswith(
        "/api/v1/health/ready', timeout=3)"
    )
    assert "version" not in compose
    assert all("container_name" not in service for service in services.values())


def test_container_files_have_no_windows_paths_secrets_or_test_dataset() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    combined = dockerfile + compose
    assert "D:\\" not in combined and "C:\\" not in combined
    assert "BTCUSDT_1h_test.parquet" not in compose
    assert "/var/run/docker.sock" not in compose
    assert "privileged:" not in compose
    assert "AS builder" in dockerfile
    assert "AS runtime" in dockerfile
    assert "AS test" in dockerfile
    assert "USER appuser" in dockerfile
    assert "pytest" not in (ROOT / "requirements-api.txt").read_text(
        encoding="utf-8"
    ).lower()
    assert ".env.*" in dockerignore
    assert "data/ml/*.parquet" in dockerignore
    assert "artifacts" in dockerignore


def test_mlflow_init_existing_registry_creates_no_versions(tmp_path: Path) -> None:
    gateway = FakeRegistryGateway({"champion": "1", "challenger": "2"})
    calls: list[str] = []

    def register(source, manifest):
        calls.append(source.alias)
        return SimpleNamespace(version="99")

    result = ensure_registry_aliases(
        gateway, _sources(tmp_path), tmp_path / "missing.json", register
    )
    assert result.aliases == {"champion": "1", "challenger": "2"}
    assert result.created_aliases == ()
    assert calls == []
    assert gateway.assignments == []


def test_mlflow_init_empty_registry_creates_both_aliases(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text('{"features": ["feature_a"]}', encoding="utf-8")
    for source in sources:
        source.model_path.write_bytes(b"fake model")
    gateway = FakeRegistryGateway()
    versions = iter(("7", "8"))
    calls: list[str] = []

    def register(source, manifest_path):
        calls.append(source.alias)
        assert manifest_path == manifest
        return SimpleNamespace(version=next(versions))

    result = ensure_registry_aliases(gateway, sources, manifest, register)
    assert result.aliases == {"champion": "7", "challenger": "8"}
    assert result.created_aliases == ("champion", "challenger")
    assert calls == ["champion", "challenger"]
    assert gateway.assignments == [("champion", "7"), ("challenger", "8")]


def test_mlflow_init_missing_artifact_fails_before_registry_change(
    tmp_path: Path,
) -> None:
    gateway = FakeRegistryGateway()
    calls: list[str] = []

    def register(source, manifest):
        calls.append(source.alias)
        return SimpleNamespace(version="1")

    try:
        ensure_registry_aliases(
            gateway, _sources(tmp_path), tmp_path / "missing.json", register
        )
    except FileNotFoundError as exc:
        assert "Cannot initialize MLflow Registry" in str(exc)
        assert "missing" in str(exc)
    else:
        raise AssertionError("Missing bootstrap artifacts must fail")
    assert calls == []
    assert gateway.assignments == []
