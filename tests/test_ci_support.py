"""Tests for the scripts and fixtures that keep CI deterministic and isolated."""

from __future__ import annotations

import json
from pathlib import Path
import re
from uuid import UUID

import joblib
import pandas as pd
import pytest
import yaml

from scripts.create_ci_bootstrap_artifacts import create_ci_bootstrap_artifacts
from scripts.smoke_test_api import build_payload
from scripts.wait_for_compose import (
    ComposeStateError,
    assess_services,
    parse_compose_ps,
    wait_until_ready,
)
from src.config import load_settings
from src.models.ci_fixture import CIConstantModel


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _ready_rows() -> list[dict[str, str | int]]:
    return [
        {"Service": "postgres", "State": "running", "Health": "healthy"},
        {"Service": "mlflow", "State": "running", "Health": "healthy"},
        {"Service": "api", "State": "running", "Health": "healthy"},
        {"Service": "migrate", "State": "exited", "ExitCode": 0},
        {"Service": "mlflow-init", "State": "exited", "ExitCode": 0},
    ]


def test_ci_fixtures_have_exact_same_28_features() -> None:
    manifest = json.loads(
        (FIXTURES / "feature_manifest.json").read_text(encoding="utf-8")
    )
    sample = json.loads(
        (FIXTURES / "inference_sample.json").read_text(encoding="utf-8")
    )
    assert len(manifest["features"]) == 28
    assert list(sample["features"]) == manifest["features"]
    assert manifest["horizon_hours"] == 3
    assert not any("target" in name or "future" in name for name in manifest["features"])


def test_ci_artifact_creation_never_fits_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CI must not fit or retrain a model")

    monkeypatch.setattr(CIConstantModel, "fit", forbidden_fit)
    baseline, challenger = create_ci_bootstrap_artifacts(
        tmp_path, FIXTURES / "feature_manifest.json"
    )
    features = pd.DataFrame([{"return_1h": 0.0}])
    baseline_model = joblib.load(baseline)
    challenger_model = joblib.load(challenger)
    assert baseline_model.predict(features).tolist() == ["HOLD"]
    assert challenger_model.predict(features).tolist() == ["SELL"]
    assert baseline_model.parameters["trained"] is False


def test_parse_compose_ps_accepts_array_and_json_lines() -> None:
    rows = _ready_rows()
    assert parse_compose_ps(json.dumps(rows)) == rows
    lines = "\n".join(json.dumps(row) for row in rows)
    assert parse_compose_ps(lines) == rows


def test_compose_ready_state_requires_health_and_successful_one_shots() -> None:
    ready, summary = assess_services(_ready_rows())
    assert ready is True
    assert "api=running/healthy" in summary
    assert "mlflow-init=exited/0" in summary


@pytest.mark.parametrize(
    "service,update,match",
    [
        ("api", {"State": "exited"}, "api entered terminal state"),
        ("mlflow", {"Health": "unhealthy"}, "mlflow is unhealthy"),
        ("migrate", {"State": "exited", "ExitCode": 2}, "exited with code 2"),
    ],
)
def test_compose_terminal_failures_stop_immediately(
    service: str, update: dict[str, str | int], match: str
) -> None:
    rows = _ready_rows()
    next(row for row in rows if row["Service"] == service).update(update)
    with pytest.raises(ComposeStateError, match=match):
        assess_services(rows)


def test_wait_for_compose_has_bounded_timeout() -> None:
    ticks = iter([0.0, 1.0, 2.0])
    reports: list[str] = []
    with pytest.raises(TimeoutError, match="did not become ready"):
        wait_until_ready(
            lambda: [],
            timeout_seconds=1.0,
            interval_seconds=0.0,
            monotonic=lambda: next(ticks),
            sleep=lambda _seconds: None,
            report=reports.append,
        )
    assert reports == ["waiting for services: postgres, mlflow, api, migrate, mlflow-init"]


def test_smoke_payload_uses_versioned_fixture_without_parquet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("CI fixture must not read an ML dataset")
        ),
    )
    settings = load_settings(ROOT / "config.yaml")
    manifest = json.loads(
        (FIXTURES / "feature_manifest.json").read_text(encoding="utf-8")
    )
    first = build_payload(
        settings,
        manifest["features"],
        validation_path=Path("forbidden.parquet"),
        sample_path=FIXTURES / "inference_sample.json",
    )
    second = build_payload(
        settings,
        manifest["features"],
        validation_path=Path("forbidden.parquet"),
        sample_path=FIXTURES / "inference_sample.json",
    )
    UUID(first["request_id"])
    assert first["request_id"] != second["request_id"]
    assert list(first["features"]) == manifest["features"]
    assert first["symbol"] == "BTCUSDT"
    assert first["timeframe"] == "1h"


def test_ci_environment_contains_disposable_values_only() -> None:
    content = (ROOT / ".env.ci.example").read_text(encoding="utf-8")
    assert "ci_password_disposable_only" in content
    assert "PAT" not in content
    assert "TOKEN" not in content
    assert "REGISTRY_BOOTSTRAP_PROFILE=ci-fixture" in content


def test_workflows_use_pinned_actions_and_least_privilege() -> None:
    ci_text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    publish_text = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    ci = yaml.load(ci_text, Loader=yaml.BaseLoader)
    publish = yaml.load(publish_text, Loader=yaml.BaseLoader)
    assert {"push", "pull_request", "workflow_dispatch", "workflow_call"} <= set(
        ci["on"]
    )
    assert "pull_request_target" not in ci["on"]
    assert ci["permissions"] == {"contents": "read"}
    assert publish["permissions"] == {"contents": "read"}
    assert publish["jobs"]["publish"]["permissions"] == {
        "contents": "read",
        "packages": "write",
    }
    assert "packages" not in publish["jobs"]["validate-manual"].get(
        "permissions", {}
    )
    all_text = ci_text + publish_text
    action_refs = re.findall(r"uses:\s+([^\s#]+)", all_text)
    remote_refs = [reference for reference in action_refs if not reference.startswith("./")]
    assert remote_refs
    assert all(re.search(r"@[0-9a-f]{40}$", reference) for reference in remote_refs)
    assert "secrets.GITHUB_TOKEN" in publish_text
    assert "GHCR_PAT" not in all_text
    assert "pull_request_target" not in all_text
    assert "+            " not in all_text


def test_publish_gate_and_ci_dependency_chain_are_explicit() -> None:
    ci = yaml.load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    publish_text = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    assert set(ci["jobs"]["docker-build"]["needs"]) == {
        "python-tests",
        "project-checks",
    }
    assert ci["jobs"]["container-tests"]["needs"] == ["docker-build"]
    assert ci["jobs"]["compose-integration"]["needs"] == ["container-tests"]
    assert "workflow_run.conclusion == 'success'" in publish_text
    assert "workflow_run.event == 'push'" in publish_text
    assert "head_branch == 'main'" in publish_text
    assert "sha-${{ env.SOURCE_SHA }}" in publish_text
    assert "type=raw,value=latest" not in publish_text
