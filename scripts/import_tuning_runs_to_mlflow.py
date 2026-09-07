"""Import saved Block 4/5 history and initialize the local Model Registry."""

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow

from src.config import load_settings
from src.mlops import MLflowTracker, ModelRegistry
from src.mlops.integration import log_buy_and_hold_run, log_tuning_runs_from_artifacts


def _latest_baseline_lightgbm_run(experiment_id: str):
    client = mlflow.MlflowClient()
    runs = client.search_runs(
        [experiment_id],
        "tags.mlflow.runName = 'baseline_lightgbm'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            "baseline_lightgbm MLflow run not found; run scripts/train_models.py first"
        )
    return runs[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(
        level=getattr(logging, settings.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    tracker = MLflowTracker(settings.mlflow)
    tracker.check_health()
    if tracker.experiment_id is None:
        raise RuntimeError("MLflow is disabled; enable mlflow.enabled to import runs")

    tuned_run_ids = log_tuning_runs_from_artifacts(
        settings, tracker, origin="imported_existing_experiment"
    )
    benchmark_run_id = log_buy_and_hold_run(settings, tracker)
    baseline_run = _latest_baseline_lightgbm_run(tracker.experiment_id)
    tuned_run = mlflow.MlflowClient().get_run(tuned_run_ids["lightgbm"])

    registry = ModelRegistry(settings.mlflow)
    baseline_version = registry.register_run_model(
        baseline_run.info.run_id,
        description=(
            "Source: Block 3 baseline LightGBM. "
            f"Validation Macro F1: {baseline_run.data.metrics['validation_macro_f1']:.6f}. "
            f"Backtest return: {baseline_run.data.metrics.get('backtest_return_pct', float('nan')):.3f}%."
        ),
        tags={
            "source_stage": "block_3",
            "model_type": "lightgbm",
            "validation_status": "passed",
            "backtest_status": "failed_profitability_check",
        },
    )
    tuned_version = registry.register_run_model(
        tuned_run.info.run_id,
        description=(
            "Source: Block 4 tuned LightGBM. "
            f"Validation Macro F1: {tuned_run.data.metrics['validation_macro_f1']:.6f}; "
            f"CV Macro F1: {tuned_run.data.metrics['cv_mean_macro_f1']:.6f} "
            f"+/- {tuned_run.data.metrics['cv_std_macro_f1']:.6f}. "
            f"Backtest return: {tuned_run.data.metrics.get('backtest_return_pct', float('nan')):.3f}%."
        ),
        tags={
            "source_stage": "block_4",
            "model_type": "lightgbm",
            "validation_status": "passed",
            "backtest_status": "failed_profitability_check",
        },
    )

    # This initial assignment is an explicit project decision, not auto-promotion.
    registry.set_champion(baseline_version.version)
    registry.set_challenger(tuned_version.version)
    print(f"Experiment ID: {tracker.experiment_id}")
    print(f"Imported tuned runs: {tuned_run_ids}")
    print(f"Buy & Hold run: {benchmark_run_id}")
    print(f"Registered model: {settings.mlflow.registered_model_name}")
    print(
        f"Champion: baseline LightGBM, version={baseline_version.version}, "
        f"run_id={baseline_run.info.run_id}"
    )
    print(
        f"Challenger: tuned LightGBM, version={tuned_version.version}, "
        f"run_id={tuned_run.info.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
