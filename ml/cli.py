"""`sentinel` -- SentinelML's command-line entry point.

Installed as a console script (see `[project.scripts]` in pyproject.toml).
Each pipeline stage (profiling now, preprocessing/training/inference in
later milestones) gets its own subcommand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.data.report import DEFAULT_REPORTS_DIR


def _run_profile(args: argparse.Namespace) -> None:
    from ml.data.report import generate_report_from_path

    outputs = generate_report_from_path(
        args.input, reports_dir=args.reports_dir, label_column=args.label_column
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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
