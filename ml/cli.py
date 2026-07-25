"""`sentinel` -- SentinelML's command-line entry point.

Installed as a console script (see `[project.scripts]` in pyproject.toml).
Each pipeline stage (profiling, preprocessing, training, and inference in
a later milestone) gets its own subcommand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.data.feature_stats import DEFAULT_LOW_VARIANCE_THRESHOLD
from ml.data.report import DEFAULT_REPORTS_DIR
from ml.preprocessing.run import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_PROCESSED_DIR,
)
from ml.preprocessing.split import DEFAULT_TEST_SIZE, DEFAULT_TRAIN_SIZE, DEFAULT_VAL_SIZE
from ml.training.run import (
    DEFAULT_MODELS_DIR,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SELECTION_METRIC,
)


def _run_profile(args: argparse.Namespace) -> None:
    from ml.data.report import generate_report_from_path

    outputs = generate_report_from_path(
        args.input, reports_dir=args.reports_dir, label_column=args.label_column
    )
    print("Generated:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


def _run_preprocess(args: argparse.Namespace) -> None:
    from ml.preprocessing.run import run_preprocessing

    outputs = run_preprocessing(
        args.input,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        label_column=args.label_column,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        low_variance_threshold=args.low_variance_threshold,
    )
    print("Generated:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


def _run_train(args: argparse.Namespace) -> None:
    from ml.training.run import run_training

    outputs = run_training(
        processed_dir=args.processed_dir,
        preprocessing_artifacts_dir=args.preprocessing_artifacts_dir,
        models_dir=args.models_dir,
        label_column=args.label_column,
        selection_metric=args.selection_metric,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    print("Generated:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


def _run_serve(args: argparse.Namespace) -> None:
    import subprocess
    import sys

    backend_dir = Path(__file__).resolve().parents[1] / "backend"
    cmd = [
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        # Our own RequestLoggingMiddleware already logs every request in
        # structured JSON; uvicorn's built-in access log would just add a
        # second, differently-formatted line per request.
        "--no-access-log",
    ]
    if args.reload:
        cmd.append("--reload")

    print(f"Starting SentinelML API on http://{args.host}:{args.port} (backend dir: {backend_dir})")
    try:
        subprocess.run(cmd, cwd=backend_dir, check=True)
    except FileNotFoundError as exc:
        print(
            "error: `uvicorn` executable not found. Run `uv sync --all-packages` "
            "to install backend dependencies.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel", description="SentinelML pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser(
        "profile", help="Generate the Milestone 1 EDA data profile report for a CSV."
    )
    profile_parser.add_argument(
        "--input", required=True, type=Path, help="Path to a CICIDS2017 CSV file"
    )
    profile_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Output directory for report artifacts (default: docs/reports)",
    )
    profile_parser.add_argument(
        "--label-column", default="Label", help="Name of the class label column"
    )
    profile_parser.set_defaults(func=_run_profile)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Run the Milestone 2 cleaning/split/encode/scale pipeline for a CSV."
    )
    preprocess_parser.add_argument(
        "--input", required=True, type=Path, help="Path to a CICIDS2017 CSV file"
    )
    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Output directory for processed train/val/test CSVs (default: ml/data/processed)",
    )
    preprocess_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Output directory for fitted imputer/scaler/label encoder/metadata "
        "(default: ml/models/artifacts/preprocessing)",
    )
    preprocess_parser.add_argument(
        "--label-column", default="Label", help="Name of the class label column"
    )
    preprocess_parser.add_argument(
        "--train-size", type=float, default=DEFAULT_TRAIN_SIZE, help="Training split fraction"
    )
    preprocess_parser.add_argument(
        "--val-size", type=float, default=DEFAULT_VAL_SIZE, help="Validation split fraction"
    )
    preprocess_parser.add_argument(
        "--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Test split fraction"
    )
    preprocess_parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for the split"
    )
    preprocess_parser.add_argument(
        "--low-variance-threshold",
        type=float,
        default=DEFAULT_LOW_VARIANCE_THRESHOLD,
        help="Normalized-variance threshold below which a feature is dropped",
    )
    preprocess_parser.set_defaults(func=_run_preprocess)

    train_parser = subparsers.add_parser(
        "train",
        help="Run the Milestone 3 training pipeline: train candidate models, "
        "evaluate, and save the best one.",
    )
    train_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing train/val/test CSVs from `sentinel preprocess` "
        "(default: ml/data/processed)",
    )
    train_parser.add_argument(
        "--preprocessing-artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing fitted preprocessing artifacts -- read for metadata only, "
        "never refit (default: ml/models/artifacts/preprocessing)",
    )
    train_parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root directory for versioned trained-model artifacts "
        "(default: ml/models/artifacts)",
    )
    train_parser.add_argument(
        "--label-column", default="Label", help="Name of the class label column"
    )
    train_parser.add_argument(
        "--selection-metric",
        default=DEFAULT_SELECTION_METRIC,
        choices=[
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
        ],
        help="Validation metric used to select the best model",
    )
    train_parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Number of trees for every candidate model",
    )
    train_parser.add_argument(
        "--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for training"
    )
    train_parser.set_defaults(func=_run_train)

    serve_parser = subparsers.add_parser(
        "serve", help="Launch the Milestone 4 FastAPI inference service locally."
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1)"
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve_parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on code changes (development only)"
    )
    serve_parser.set_defaults(func=_run_serve)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
