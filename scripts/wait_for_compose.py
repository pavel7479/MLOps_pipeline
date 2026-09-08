"""Wait for long-running and one-shot Docker Compose services."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import time
from typing import Any

LONG_RUNNING = ("postgres", "mlflow", "api")
ONE_SHOT = ("migrate", "mlflow-init")


class ComposeStateError(RuntimeError):
    """A service entered a terminal state that cannot become ready."""


def parse_compose_ps(output: str) -> list[dict[str, Any]]:
    """Accept both JSON-array and JSON-lines output produced by Compose v2."""
    stripped = output.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return [json.loads(line) for line in stripped.splitlines() if line.strip()]
    return parsed if isinstance(parsed, list) else [parsed]


def assess_services(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return readiness and a concise state summary, or raise on hard failure."""
    services = {str(row.get("Service")): row for row in rows}
    required = (*LONG_RUNNING, *ONE_SHOT)
    missing = [service for service in required if service not in services]
    if missing:
        return False, f"waiting for services: {', '.join(missing)}"

    details: list[str] = []
    ready = True
    for service in LONG_RUNNING:
        row = services[service]
        state = str(row.get("State", "")).lower()
        health = str(row.get("Health", "")).lower()
        details.append(f"{service}={state}/{health or 'no-health'}")
        if state in {"dead", "exited", "removing"}:
            raise ComposeStateError(f"{service} entered terminal state {state}")
        if health == "unhealthy":
            raise ComposeStateError(f"{service} is unhealthy")
        ready = ready and state == "running" and health == "healthy"

    for service in ONE_SHOT:
        row = services[service]
        state = str(row.get("State", "")).lower()
        exit_code = int(row.get("ExitCode") or 0)
        details.append(f"{service}={state}/{exit_code}")
        if state == "exited" and exit_code != 0:
            raise ComposeStateError(f"{service} exited with code {exit_code}")
        ready = ready and state == "exited" and exit_code == 0
    return ready, ", ".join(details)


def wait_until_ready(
    fetch: Callable[[], list[dict[str, Any]]],
    *,
    timeout_seconds: float,
    interval_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    report: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Poll actual Compose state until ready or a bounded timeout expires."""
    deadline = monotonic() + timeout_seconds
    previous = ""
    while True:
        rows = fetch()
        ready, summary = assess_services(rows)
        if summary != previous:
            report(summary)
            previous = summary
        if ready:
            return rows
        if monotonic() >= deadline:
            raise TimeoutError(
                f"Compose services did not become ready in {timeout_seconds:g}s; "
                f"last state: {summary}"
            )
        sleep(interval_seconds)


def compose_rows(project_name: str, env_file: Path) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--project-name",
            project_name,
            "--env-file",
            str(env_file),
            "ps",
            "--all",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_compose_ps(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", default="crypto-ml-ci")
    parser.add_argument("--env-file", type=Path, default=Path(".env.ci.example"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    wait_until_ready(
        lambda: compose_rows(args.project_name, args.env_file),
        timeout_seconds=args.timeout,
        interval_seconds=args.interval,
    )
    print("Compose stack is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
