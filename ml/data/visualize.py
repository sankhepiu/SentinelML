"""Visualizations for the CICIDS2017 data profile.

Every function here reads `df` and writes a PNG -- nothing is mutated,
cleaned, or dropped from the underlying DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set_theme(style="whitegrid")

MISSING_MATRIX_SAMPLE_ROWS = 2000
TOP_N_NUMERIC_FEATURES = 9
CORRELATION_HEATMAP_MAX_FEATURES = 60


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_class_distribution(
    df: pd.DataFrame, label_column: str, output_path: str | Path
) -> Path | None:
    """Bar chart of row counts per class, log-scaled (CICIDS2017 is heavily imbalanced)."""
    if label_column not in df.columns:
        return None
    counts = df[label_column].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.6), 6))
    labels = counts.index.astype(str)
    sns.barplot(x=labels, y=counts.values, hue=labels, palette="viridis", legend=False, ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Row count (log scale)")
    ax.set_xlabel(label_column)
    ax.set_title("Class distribution")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, output_path)


def plot_missing_value_matrix(
    df: pd.DataFrame, output_path: str | Path, *, sample_rows: int = MISSING_MATRIX_SAMPLE_ROWS
) -> Path:
    """Boolean heatmap of missing values, row-sampled for readability on large datasets."""
    sample = df if len(df) <= sample_rows else df.sample(sample_rows, random_state=42)
    missing = sample.isna()
    fig, ax = plt.subplots(figsize=(min(24, max(10, df.shape[1] * 0.25)), 8))
    sns.heatmap(
        missing,
        cbar=False,
        yticklabels=False,
        cmap=ListedColormap(["#3182bd", "#d73027"]),
        ax=ax,
    )
    ax.set_title(f"Missing value matrix (sample of {len(sample):,} of {len(df):,} rows)")
    ax.set_xlabel("Column")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    return _save(fig, output_path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    max_features: int = CORRELATION_HEATMAP_MAX_FEATURES,
) -> Path | None:
    """Pearson correlation heatmap over numeric columns.

    If there are more than `max_features` numeric columns, keeps the
    highest-variance ones so the heatmap stays legible.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None
    clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    if len(numeric_cols) > max_features:
        numeric_cols = clean.var().sort_values(ascending=False).head(max_features).index.tolist()
        clean = clean[numeric_cols]
    corr = clean.corr(numeric_only=True)
    fig, ax = plt.subplots(
        figsize=(max(10, len(numeric_cols) * 0.35), max(8, len(numeric_cols) * 0.3))
    )
    sns.heatmap(
        corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson r"}
    )
    ax.set_title("Feature correlation heatmap")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    return _save(fig, output_path)


def plot_top_numeric_distributions(
    df: pd.DataFrame, output_path: str | Path, *, n: int = TOP_N_NUMERIC_FEATURES
) -> Path | None:
    """Histogram grid for the `n` numeric columns with the highest normalized variance."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    normalized_variances: dict[str, float] = {}
    for col in numeric_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty or series.max() == series.min():
            continue
        normalized = (series - series.min()) / (series.max() - series.min())
        normalized_variances[col] = float(normalized.var())

    top_cols = sorted(normalized_variances, key=normalized_variances.get, reverse=True)[:n]
    if not top_cols:
        return None

    n_grid_cols = 3
    n_grid_rows = -(-len(top_cols) // n_grid_cols)
    fig, axes = plt.subplots(n_grid_rows, n_grid_cols, figsize=(5 * n_grid_cols, 4 * n_grid_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, top_cols, strict=False):
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        sns.histplot(data, bins=50, ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for ax in axes[len(top_cols) :]:
        ax.axis("off")
    fig.suptitle("Top numerical feature distributions (by normalized variance)")
    return _save(fig, output_path)


def generate_all_visualizations(
    df: pd.DataFrame, output_dir: str | Path, *, label_column: str = "Label"
) -> dict[str, Path | None]:
    """Generate every required figure and save it under `output_dir`. Returns paths by name."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "class_distribution": plot_class_distribution(
            df, label_column, output_dir / "class_distribution.png"
        ),
        "missing_value_matrix": plot_missing_value_matrix(
            df, output_dir / "missing_value_matrix.png"
        ),
        "correlation_heatmap": plot_correlation_heatmap(df, output_dir / "correlation_heatmap.png"),
        "top_numerical_feature_distributions": plot_top_numeric_distributions(
            df, output_dir / "top_numerical_feature_distributions.png"
        ),
    }
