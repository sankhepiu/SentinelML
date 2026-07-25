"""`sentinel` -- SentinelML's command-line entry point.

Installed as a console script (see `[project.scripts]` in pyproject.toml).
Each pipeline stage (profiling, preprocessing, and training/inference in
later milestones) gets its own subcommand.
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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
