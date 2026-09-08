"""Exercise the real API with one Validation row and verify persistence."""

import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import pandas as pd

from src.config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config)
    base_url = args.base_url or os.getenv("API_BASE_URL") or (
        f"http://{settings.api.host}:{settings.api.port}"
    )
    stem = f"{settings.market_data.symbol}_{settings.market_data.timeframe}"
    validation_path = Path(
        os.getenv(
            "VALIDATION_DATA_PATH",
            str(settings.ml_dataset.output_dir / f"{stem}_validation.parquet"),
        )
    )
    manifest_path = Path(
        os.getenv(
            "FEATURE_MANIFEST_PATH",
            str(settings.ml_dataset.output_dir / "feature_manifest.json"),
        )
    )
    feature_names = json.loads(manifest_path.read_text(encoding="utf-8"))["features"]
    validation = pd.read_parquet(validation_path)
    row = validation.iloc[0]
    request_id = str(uuid4())
    payload = {
        "request_id": request_id,
        "symbol": settings.inference.expected_symbol,
        "timeframe": settings.inference.expected_timeframe,
        "feature_timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "features": {name: float(row[name]) for name in feature_names},
    }

    with httpx.Client(base_url=base_url, timeout=30) as client:
        live = client.get("/api/v1/health/live")
        ready = client.get("/api/v1/health/ready")
        model = client.get("/api/v1/model")
        created = client.post("/api/v1/predict", json=payload)
        stored = client.get(f"/api/v1/predictions/{request_id}")
        replay = client.post("/api/v1/predict", json=payload)
        conflict_payload = {
            **payload,
            "features": {
                **payload["features"],
                feature_names[0]: payload["features"][feature_names[0]] + 0.000001,
            },
        }
        conflict = client.post("/api/v1/predict", json=conflict_payload)
        for response in (live, ready, model, created, stored, replay):
            response.raise_for_status()
        if conflict.status_code != 409:
            raise RuntimeError(
                f"Expected idempotency conflict 409, got {conflict.status_code}"
            )

    model_data = model.json()
    created_data = created.json()
    stored_data = stored.json()
    replay_data = replay.json()
    if created_data["prediction"] != stored_data["prediction"]:
        raise RuntimeError("Stored prediction differs from POST response")
    if created_data["replayed"] or not replay_data["replayed"]:
        raise RuntimeError("Idempotency replay flags are invalid")
    print(f"API health: {live.json()['status'].upper()}")
    print(f"Database: {ready.json()['database'].upper()}")
    print(
        "Loaded model: "
        f"{model_data['registered_name']}@{model_data['alias']} "
        f"version={model_data['version']} run_id={model_data['run_id']}"
    )
    print(f"Feature count: {model_data['features_count']}")
    print(
        f"POST /api/v1/predict: {created.status_code} "
        f"prediction={created_data['prediction']} request_id={request_id}"
    )
    print(
        f"PostgreSQL persistence: confirmed, "
        f"created_at={stored_data['created_at']}"
    )
    print(
        f"Idempotency: first={created_data['replayed']} "
        f"replay={replay_data['replayed']} conflict={conflict.status_code}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
