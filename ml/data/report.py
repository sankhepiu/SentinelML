"""Milestone 1 EDA pipeline: load -> profile -> visualize -> markdown report.

Strictly read-only exploratory analysis: the dataset is never cleaned,
imputed, or used for training. Run via the `sentinel` CLI once the target
CSV is in place under `ml/data/raw/` (see `ml/data/README.md`):

    uv run sentinel profile --input ml/data/raw/Wednesday-workingHours.pcap_ISCX.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.data.loader import (
    DEFAULT_DATASET,
    RAW_DATA_DIR,
    load_cicids_csv,
    resolve_dataset_path,
)
from ml.data.profile import DataProfile, generate_profile
from ml.data.visualize import generate_all_visualizations

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = REPO_ROOT / "docs" / "reports"

_FIGURE_TITLES = {
    "class_distribution": "Class Distribution",
    "missing_value_matrix": "Missing Value Matrix",
    "correlation_heatmap": "Correlation Heatmap",
    "top_numerical_feature_distributions": "Top Numerical Feature Distributions",
}


def _format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def _write_summary_statistics_csv(profile: DataProfile, path: Path) -> Path:
    pd.DataFrame(profile.summary_statistics).to_csv(path)
    return path


def _write_correlation_pairs_csv(profile: DataProfile, path: Path) -> Path:
    pd.DataFrame(profile.highly_correlated_pairs).to_csv(path, index=False)
    return path


def _data_quality_section(profile: DataProfile) -> str:
    lines = ["## Observed Data Quality Issues", ""]

    if profile.missing_values:
        lines.append(
            f"- **Missing values**: {len(profile.missing_values)} column(s) contain nulls."
        )
        worst = sorted(profile.missing_values.items(), key=lambda kv: kv[1]["count"], reverse=True)
        for col, stats in worst[:5]:
            lines.append(f"  - `{col}`: {stats['count']:,} missing ({stats['pct']}%)")
    else:
        lines.append("- **Missing values**: none observed.")

    if profile.infinity_values:
        lines.append(
            f"- **Infinite values**: {len(profile.infinity_values)} numeric column(s) contain ±Inf."
        )
        for col, count in sorted(
            profile.infinity_values.items(), key=lambda kv: kv[1], reverse=True
        ):
            lines.append(f"  - `{col}`: {count:,} infinite value(s)")
        lines.append(
            "  - These typically arise from rate features (e.g. bytes/s, packets/s) computed "
            "by dividing by a zero flow duration."
        )
    else:
        lines.append("- **Infinite values**: none observed.")

    lines.append(
        f"- **Duplicate rows**: {profile.duplicate_row_count:,} "
        f"({profile.duplicate_row_pct}% of all rows)."
    )

    if profile.constant_columns:
        cols = ", ".join(f"`{c}`" for c in profile.constant_columns)
        lines.append(f"- **Constant columns** (zero information content): {cols}")
    else:
        lines.append("- **Constant columns**: none observed.")

    if profile.low_variance_columns:
        cols = ", ".join(f"`{c}`" for c in profile.low_variance_columns)
        lines.append(f"- **Low-variance columns** (normalized variance below threshold): {cols}")
    else:
        lines.append("- **Low-variance columns**: none observed.")

    if profile.highly_correlated_pairs:
        lines.append(
            f"- **Highly correlated feature pairs**: {len(profile.highly_correlated_pairs)} "
            "pair(s) at |r| ≥ threshold (full list in `correlation_pairs.csv`). Top 5:"
        )
        for pair in profile.highly_correlated_pairs[:5]:
            lines.append(
                f"  - `{pair['feature_a']}` <-> `{pair['feature_b']}`: r = {pair['correlation']}"
            )
    else:
        lines.append("- **Highly correlated feature pairs**: none observed.")

    if profile.class_distribution:
        counts = sorted(
            profile.class_distribution.items(), key=lambda kv: kv[1]["count"], reverse=True
        )
        majority_label, majority_stats = counts[0]
        minority_label, minority_stats = counts[-1]
        ratio = (
            majority_stats["count"] / minority_stats["count"]
            if minority_stats["count"]
            else float("inf")
        )
        lines.append(
            f"- **Class imbalance**: {len(counts)} class(es); majority class `{majority_label}` "
            f"is {ratio:,.1f}x larger than minority class `{minority_label}`."
        )

    return "\n".join(lines)


def _preprocessing_steps_section(profile: DataProfile) -> str:
    steps: list[str] = []

    if profile.constant_columns:
        steps.append(
            f"Drop constant columns ({len(profile.constant_columns)}) -- they carry no "
            "predictive signal and only add dimensionality."
        )
    if profile.infinity_values:
        cols = ", ".join(f"`{c}`" for c in profile.infinity_values)
        steps.append(
            f"Replace ±Inf in rate-derived columns ({cols}) with NaN, then apply the same "
            "missing-value strategy used for the rest of the dataset."
        )
    if profile.missing_values:
        steps.append(
            "Impute or drop rows/columns with missing values, depending on how concentrated "
            "the missingness is per column (see `data_profile.json` for the per-column breakdown)."
        )
    if profile.duplicate_row_count:
        steps.append(
            f"Deduplicate the {profile.duplicate_row_count:,} exact-duplicate rows before any "
            "train/test split, to avoid leakage between splits."
        )
    if profile.low_variance_columns:
        cols = ", ".join(f"`{c}`" for c in profile.low_variance_columns)
        steps.append(
            f"Review low-variance columns ({cols}) for removal -- they add little separability "
            "but do add training cost."
        )
    if profile.highly_correlated_pairs:
        steps.append(
            "Reduce redundancy among highly correlated feature pairs (drop one from each pair, "
            "or apply dimensionality reduction) to limit multicollinearity."
        )
    steps.append(
        "Encode the categorical label column and scale/normalize numeric features -- flow "
        "features span very different units and magnitudes (durations in microseconds vs. "
        "byte/packet counts)."
    )
    if profile.class_distribution and len(profile.class_distribution) > 1:
        steps.append(
            "Address class imbalance (e.g. class weighting, oversampling minority attack "
            "classes, or stratified sampling) before training."
        )

    lines = ["## Expected Preprocessing Steps", ""]
    lines += [f"{i}. {step}" for i, step in enumerate(steps, start=1)]
    return "\n".join(lines)


def _modelling_challenges_section(profile: DataProfile) -> str:
    lines = ["## Potential Modelling Challenges", ""]

    if profile.class_distribution and len(profile.class_distribution) > 1:
        counts = sorted(
            profile.class_distribution.items(), key=lambda kv: kv[1]["count"], reverse=True
        )
        majority_stats, minority_stats = counts[0][1], counts[-1][1]
        ratio = (
            majority_stats["count"] / minority_stats["count"]
            if minority_stats["count"]
            else float("inf")
        )
        lines.append(
            f"- **Severe class imbalance** ({ratio:,.1f}x majority/minority ratio): accuracy "
            "alone will be a misleading metric; rare attack classes risk being ignored by the "
            "model unless explicitly weighted or resampled."
        )

    if profile.highly_correlated_pairs:
        lines.append(
            f"- **Multicollinearity**: {len(profile.highly_correlated_pairs)} feature pair(s) "
            "are highly correlated, which can destabilize linear/coefficient-based models and "
            "inflate feature-importance estimates for tree-based models."
        )

    lines.append(
        f"- **Dimensionality vs. row count**: {profile.n_cols} columns over "
        f"{profile.n_rows:,} rows ({_format_bytes(profile.memory_usage_bytes)} in memory) -- "
        "feature selection or dimensionality reduction may be needed to control training cost "
        "and overfitting risk."
    )

    if profile.infinity_values or profile.missing_values:
        lines.append(
            "- **Data integrity artifacts** (±Inf/NaN in rate-based features) must be resolved "
            "consistently between training and inference, or the serving pipeline will crash or "
            "silently diverge from what the model was trained on."
        )

    lines.append(
        "- **Generalization beyond this capture window**: CICIDS2017 reflects one lab's traffic "
        "and attack patterns over a fixed time window; models trained on it may not transfer to "
        "production traffic without further validation."
    )

    return "\n".join(lines)


def _render_markdown_report(
    profile: DataProfile,
    dataset_name: str,
    source_path: Path,
    figures: dict[str, Path | None],
) -> str:
    figures_md = []
    for name, path in figures.items():
        if path is None:
            continue
        title = _FIGURE_TITLES.get(name, name.replace("_", " ").title())
        figures_md.append(f"### {title}\n\n![{title}](figures/{path.name})")

    return "\n\n".join(
        [
            "# CICIDS2017 Data Profile Report",
            "",
            f"**Dataset:** `{dataset_name}` ({source_path.name})  \n"
            f"**Rows:** {profile.n_rows:,}  **Columns:** {profile.n_cols}  "
            f"**Memory usage:** {_format_bytes(profile.memory_usage_bytes)}",
            "This report is generated by `ml/data/report.py` and is strictly exploratory: the "
            "underlying dataset is never cleaned, modified, or used for training. Full numeric "
            "detail lives alongside this file in `data_profile.json`, `summary_statistics.csv`, "
            "and `correlation_pairs.csv`.",
            _data_quality_section(profile),
            _preprocessing_steps_section(profile),
            _modelling_challenges_section(profile),
            "## Visualizations",
            "\n\n".join(figures_md) if figures_md else "_No visualizations were generated._",
            "",
        ]
    )


def generate_report_from_path(
    source_path: str | Path,
    *,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    label_column: str = "Label",
    dataset_label: str | None = None,
) -> dict[str, Path]:
    """Run the full M1 pipeline against a CSV at `source_path`.

    Never mutates, cleans, or trains on the input -- read-only exploratory
    analysis. `dataset_label` is cosmetic (used in the report heading);
    defaults to the file's stem.
    """
    source_path = Path(source_path)
    df = load_cicids_csv(source_path)

    profile = generate_profile(df, label_column=label_column)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures = generate_all_visualizations(df, reports_dir / "figures", label_column=label_column)

    outputs = {
        "report": None,  # filled in below, after the markdown body is rendered
        "profile_json": profile.to_json(reports_dir / "data_profile.json"),
        "summary_statistics_csv": _write_summary_statistics_csv(
            profile, reports_dir / "summary_statistics.csv"
        ),
        "correlation_pairs_csv": _write_correlation_pairs_csv(
            profile, reports_dir / "correlation_pairs.csv"
        ),
    }
    outputs.update({f"figure_{name}": path for name, path in figures.items() if path is not None})

    report_markdown = _render_markdown_report(
        profile, dataset_label or source_path.stem, source_path, figures
    )
    report_path = reports_dir / "data_profile_report.md"
    report_path.write_text(report_markdown)
    outputs["report"] = report_path

    return outputs


def generate_report(
    dataset_name: str = DEFAULT_DATASET,
    *,
    raw_dir: Path = RAW_DATA_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    label_column: str = "Label",
) -> dict[str, Path]:
    """Resolve `dataset_name` to a file under `raw_dir` and run the M1 pipeline on it.

    Raises `DatasetNotFoundError` if the CSV isn't present.
    """
    source_path = resolve_dataset_path(dataset_name, raw_dir)
    return generate_report_from_path(
        source_path, reports_dir=reports_dir, label_column=label_column, dataset_label=dataset_name
    )
