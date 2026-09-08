"""Create tiny deterministic Registry artifacts for a disposable CI stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ci_fixture import CIConstantModel


def create_ci_bootstrap_artifacts(
    output_root: Path, manifest_path: Path
) -> tuple[Path, Path]:
    """Serialize two test-only models without fitting or reading a dataset."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest.get("features")
    if (
        not isinstance(features, list)
        or len(features) != 28
        or not all(isinstance(name, str) and name for name in features)
    ):
        raise ValueError("CI feature manifest must contain exactly 28 feature names")

    baseline = output_root / "models" / "lightgbm.joblib"
    challenger = output_root / "tuned_models" / "lightgbm_tuned.joblib"
    CIConstantModel("HOLD").save(baseline)
    CIConstantModel("SELL").save(challenger)
    return baseline, challenger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    args = parser.parse_args()
    baseline, challenger = create_ci_bootstrap_artifacts(
        args.output_root, args.feature_manifest
    )
    print("Created deterministic CI Registry fixtures without model training:")
    print(f"  champion source: {baseline}")
    print(f"  challenger source: {challenger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
