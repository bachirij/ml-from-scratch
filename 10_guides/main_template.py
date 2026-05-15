"""
<project-name> - Pipeline Orchestrator
=======================================

Single entry point to run all or part of the ML pipeline.

Usage:
------
    # Run all steps
    python main.py

    # Run a specific step
    python main.py --steps ingest
    python main.py --steps train
    python main.py --steps evaluate

    # Chain multiple steps
    python main.py --steps preprocess features train

    # Run with custom parameters
    python main.py --steps ingest preprocess --start-date 2023-01-01 --end-date 2024-12-31

    # Run real-time ingestion manually
    python main.py --steps realtime

    # Run drift monitoring manually
    python main.py --steps monitor

    # Display help
    python main.py --help

Pipeline steps:
---------------
    ingest      → fetch raw data from external sources
    preprocess  → clean, merge, validate raw data
    features    → feature engineering (lags, encodings, transformations)
    train       → train models, select best, save artifacts
    evaluate    → full evaluation on test set, generate reports
    realtime    → one manual real-time ingestion + prediction cycle
    monitor     → drift detection (features + predictions vs. reference)
    retrain     → retrain from scratch when drift is detected

Notes:
------
- Steps ingest / preprocess / features use --start-date and --end-date.
- Steps train / evaluate ignore date args: temporal splits are managed
  by src/modeling/config.py.
- Step realtime ignores date args (fetches recent window only).
- Guard clauses in each step skip already-existing files automatically.
  Delete the relevant files manually to force reprocessing.
"""

import argparse
import sys
import time
from datetime import datetime


# ---------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------

# Add any project-level constants here.
# Examples: target entity (country, store, sensor), API endpoints, paths.

CONFIG = {
    "entity": "default",          # adapt to your project (country, site, model_id…)
    "raw_data_path": "data/raw/",
    "processed_data_path": "data/processed/",
    "features_path": "data/features/",
    "models_path": "models/",
    "reports_path": "reports/",
    "monitoring_path": "data/monitoring/",
}

ALL_STEPS = [
    "ingest",
    "preprocess",
    "features",
    "train",
    "evaluate",
    "realtime",
    "monitor",
    "retrain",
]

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE   = str(datetime.now().year - 1) + "-12-31"


# ---------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------

def import_modules() -> dict:
    """
    Import all pipeline modules.

    Centralised here to surface clear error messages if a module
    or dependency is missing, instead of a raw Python traceback.
    Add or remove imports to match your project structure.
    """
    try:
        from src.ingestion.ingest import fetch_and_store                        # noqa
        from src.preprocessing.preprocess import build_processed_dataset        # noqa
        from src.feature_engineering.build_features import build_features       # noqa
        from src.modeling.train import run_training                              # noqa
        from src.modeling.evaluate import run_evaluation                        # noqa
        from src.ingestion.realtime import fetch_and_store_realtime             # noqa
        from src.monitoring.monitor import run_monitoring                       # noqa
        from src.modeling.retrain import run_retraining                         # noqa

        return {
            "fetch_and_store":          fetch_and_store,
            "build_processed_dataset":  build_processed_dataset,
            "build_features":           build_features,
            "run_training":             run_training,
            "run_evaluation":           run_evaluation,
            "fetch_and_store_realtime": fetch_and_store_realtime,
            "run_monitoring":           run_monitoring,
            "run_retraining":           run_retraining,
        }

    except ImportError as e:
        print(f"\n[ERROR] Failed to import a module: {e}")
        print("  → Make sure you run main.py from the project root.")
        print("  → Example: python main.py --steps ingest\n")
        sys.exit(1)


# ---------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------

def step_ingest(config: dict, start_date: str, end_date: str, modules: dict):
    """
    Step 1 — Raw data ingestion.
    Fetches raw data from external sources (APIs, databases, files)
    and stores it locally.
    Already-existing files are skipped automatically (guard clause in module).

    Adapt: add source-specific parameters (API keys, endpoints, entity ids).
    """
    print("\n--- Raw data ingestion ---")
    modules["fetch_and_store"](
        entity=config["entity"],
        start_date=start_date,
        end_date=end_date,
        output_path=config["raw_data_path"],
    )


def step_preprocess(config: dict, start_date: str, end_date: str, modules: dict):
    """
    Step 2 — Preprocessing.
    Cleans, reindexes, interpolates and merges raw sources.
    Outputs a single processed dataset per entity / time window.

    Adapt: add source names, join keys, interpolation strategy.
    """
    print("\n--- Preprocessing ---")
    modules["build_processed_dataset"](
        entity=config["entity"],
        start_date=start_date,
        end_date=end_date,
        input_path=config["raw_data_path"],
        output_path=config["processed_data_path"],
    )


def step_features(config: dict, start_date: str, end_date: str, modules: dict):
    """
    Step 3 — Feature engineering.
    Builds all model-ready features (lags, rolling stats, calendar,
    target encoding, etc.) from the processed dataset.

    Adapt: add forecast horizon, feature groups, encoding config.
    """
    print("\n--- Feature engineering ---")
    modules["build_features"](
        entity=config["entity"],
        start_date=start_date,
        end_date=end_date,
        input_path=config["processed_data_path"],
        output_path=config["features_path"],
    )


def step_train(config: dict, start_date: str, end_date: str, modules: dict):
    """
    Step 4 — Model training.
    Trains one or several models, selects the best, saves artifacts.
    Temporal splits are managed by src/modeling/config.py —
    start_date / end_date are intentionally ignored here.

    Adapt: add model list, scoring metric, cv strategy.
    """
    print("\n--- Model training ---")
    print("    [INFO] Temporal splits are managed by src/modeling/config.py")
    modules["run_training"](
        entity=config["entity"],
        features_path=config["features_path"],
        output_path=config["models_path"],
    )


def step_evaluate(config: dict, start_date: str, end_date: str, modules: dict):
    """
    Step 5 — Evaluation.
    Runs the saved model on the held-out test set.
    Generates a full evaluation report (metrics, plots, confusion matrix).
    start_date / end_date are ignored — test set is fixed by config.

    Adapt: add metric list, report format (HTML, JSON, PDF).
    """
    print("\n--- Model evaluation ---")
    modules["run_evaluation"](
        entity=config["entity"],
        models_path=config["models_path"],
        features_path=config["features_path"],
        output_path=config["reports_path"],
    )


def step_realtime(config: dict, start_date: str, end_date: str, modules: dict):
    """
    Step 6 — Real-time ingestion (manual trigger).
    Fetches the most recent data window, runs the model, stores predictions.
    Automatically triggers drift monitoring after ingestion.
    start_date / end_date are ignored (window defined in realtime module).

    For automated execution (cron, scheduler), use scheduler.py instead.
    Adapt: set recent window size, prediction horizon.
    """
    print("\n--- Real-time ingestion and prediction ---")
    modules["fetch_and_store_realtime"](
        entity=config["entity"],
        models_path=config["models_path"],
        output_path=config["raw_data_path"],
    )

    print("\n--- Drift monitoring (auto-triggered after realtime) ---")
    modules["run_monitoring"](
        entity=config["entity"],
        monitoring_path=config["monitoring_path"],
    )


def step_monitor(config: dict, start_date: str, end_date: str, modules: dict):
    """
    Step 7 — Drift monitoring (manual trigger).
    Compares current feature distributions and prediction distributions
    against the training reference using statistical tests.
    Saves a timestamped report and logs [WARN] alerts on critical features.
    Date args are ignored.

    Adapt: set reference dataset, drift thresholds, alert channels.
    """
    print("\n--- Drift monitoring ---")
    modules["run_monitoring"](
        entity=config["entity"],
        monitoring_path=config["monitoring_path"],
    )


def step_retrain(config: dict, start_date: str, end_date: str, modules: dict):
    """
    Step 8 — Retraining.
    Full retrain from scratch on the most recent data window.
    Triggered manually or automatically when monitoring detects drift.
    Date args are ignored — retraining window is managed by config.

    Adapt: set retraining frequency, warm-start vs. cold-start strategy.
    """
    print("\n--- Model retraining ---")
    modules["run_retraining"](
        entity=config["entity"],
        features_path=config["features_path"],
        models_path=config["models_path"],
    )


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

STEP_FUNCTIONS = {
    "ingest":     step_ingest,
    "preprocess": step_preprocess,
    "features":   step_features,
    "train":      step_train,
    "evaluate":   step_evaluate,
    "realtime":   step_realtime,
    "monitor":    step_monitor,
    "retrain":    step_retrain,
}


def run_pipeline(steps: list, config: dict, start_date: str, end_date: str):
    """
    Execute the requested pipeline steps in order.

    Parameters
    ----------
    steps : list[str]
        Ordered list of steps to run (subset of ALL_STEPS).
    config : dict
        Project-level configuration (paths, entity, constants).
    start_date : str
        Start date for data steps (ISO format: YYYY-MM-DD).
    end_date : str
        End date for data steps, inclusive.
    """
    modules = import_modules()

    print("=" * 60)
    print("  Pipeline Orchestrator")
    print("=" * 60)
    print(f"  Entity     : {config['entity']}")
    print(f"  Date range : {start_date} → {end_date}")
    print(f"  Steps      : {' → '.join(steps)}")
    print(f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pipeline_start = time.time()

    for i, step_name in enumerate(steps, start=1):
        print(f"\n{'=' * 60}")
        print(f"  STEP {i}/{len(steps)} : {step_name.upper()}")
        print(f"{'=' * 60}")

        step_start = time.time()

        STEP_FUNCTIONS[step_name](
            config=config,
            start_date=start_date,
            end_date=end_date,
            modules=modules,
        )

        elapsed = time.time() - step_start
        print(f"\n[OK] '{step_name}' completed in {elapsed:.1f}s")

    total = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  All steps completed in {total:.1f}s")
    print(f"  Finished at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML Pipeline Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        Examples:
        python main.py                                         # all steps, default dates
        python main.py --steps ingest                          # ingestion only
        python main.py --steps preprocess features             # preprocess + features
        python main.py --steps train evaluate                  # train then evaluate
        python main.py --steps realtime                        # manual real-time cycle
        python main.py --steps monitor                         # manual drift check
        python main.py --steps retrain                         # force full retrain
        python main.py --steps ingest preprocess \\
                        --start-date 2024-01-01 \\
                        --end-date 2024-12-31               # single year
        """,
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=ALL_STEPS + ["all"],
        metavar="STEP",
        help=(
            "Steps to run (default: all). Choices:\n"
            "  ingest      → fetch raw data from external sources\n"
            "  preprocess  → clean and merge raw data\n"
            "  features    → build model-ready features\n"
            "  train       → train models and save artifacts\n"
            "  evaluate    → evaluate on test set, generate report\n"
            "  realtime    → manual real-time ingestion + prediction cycle\n"
            "  monitor     → drift detection vs. training reference\n"
            "  retrain     → full retrain when drift is detected\n"
            "  all         → run all steps in order\n"
        ),
    )

    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        dest="start_date",
        help=f"Start date for data steps, ISO format (default: {DEFAULT_START_DATE})",
    )

    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        dest="end_date",
        help=f"End date for data steps, inclusive, ISO format (default: {DEFAULT_END_DATE})",
    )

    # Optional: add project-specific arguments here.
    # Example: --entity, --model, --env (dev/staging/prod)

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    steps = ALL_STEPS if "all" in args.steps else args.steps

    # Validate date format
    for date_str, label in [(args.start_date, "--start-date"), (args.end_date, "--end-date")]:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"[ERROR] {label} must be in ISO format YYYY-MM-DD, got: '{date_str}'")
            sys.exit(1)

    if args.start_date > args.end_date:
        print(f"[ERROR] --start-date ({args.start_date}) must be <= --end-date ({args.end_date})")
        sys.exit(1)

    run_pipeline(
        steps=steps,
        config=CONFIG,
        start_date=args.start_date,
        end_date=args.end_date,
    )