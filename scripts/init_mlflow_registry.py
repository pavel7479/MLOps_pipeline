"""Check or restore champion/challenger aliases for the container stack."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.mlops.registry_bootstrap import initialize_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=Path("/bootstrap/models/lightgbm.joblib"),
    )
    parser.add_argument(
        "--tuned-model",
        type=Path,
        default=Path("/bootstrap/tuned_models/lightgbm_tuned.joblib"),
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("/bootstrap/feature_manifest.json"),
    )
    parser.add_argument(
        "--bootstrap-profile",
        choices=("saved-models", "ci-fixture"),
        default=os.getenv("REGISTRY_BOOTSTRAP_PROFILE", "saved-models"),
        help="Use saved project models locally or deterministic fixtures in CI.",
    )
    args = parser.parse_args()
    result = initialize_registry(
        load_settings(args.config),
        baseline_model_path=args.baseline_model,
        tuned_model_path=args.tuned_model,
        manifest_path=args.feature_manifest,
        bootstrap_profile=args.bootstrap_profile,
    )
    created = ", ".join(result.created_aliases) or "none"
    print(
        "MLflow Registry ready: "
        f"champion={result.aliases['champion']}, "
        f"challenger={result.aliases['challenger']}, created={created}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
