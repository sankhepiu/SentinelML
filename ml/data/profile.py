"""Data profiling for CICIDS2017 -- read-only statistics, no data mutation.

`generate_profile` computes every metric Milestone 1 requires from a
DataFrame as loaded by `ml.data.loader`. It never modifies, cleans, or casts
the input DataFrame -- profiling only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_LOW_VARIANCE_THRESHOLD = 0.01
DEFAULT_CORRELATION_THRESHOLD = 0.95


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    memory_usage_bytes: int
    column_dtypes: dict[str, str]
    missing_values: dict[str, dict[str, float]]
    infinity_values: dict[str, int]
    duplicate_row_count: int
    duplicate_row_pct: float
    class_distribution: dict[str, dict[str, float]] | None
    summary_statistics: dict[str, dict[str, float]]
    constant_columns: list[str]
    low_variance_columns: dict[str, float]
    highly_correlated_pairs: list[dict[str, Any]]
    label_column: str | None

    def to_dict(self) -> dict[str, Any]:
        return _to_native(asdict(self))

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=indent))
        return path


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy scalar types (and NaN) to native JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _find_constant_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


def _find_low_variance_columns(
    df: pd.DataFrame, numeric_cols: list[str], exclude: set[str], threshold: float
) -> dict[str, float]:
    """Min-max normalized variance below `threshold`, excluding constant columns.

    Raw variance is scale-dependent (a column measured in bytes will always
    look "higher variance" than one measured in seconds), so columns are
    normalized to [0, 1] before comparing.
    """
    low_variance = {}
    for col in numeric_cols:
        if col in exclude:
            continue
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        col_min, col_max = series.min(), series.max()
        if col_max == col_min:
            continue
        normalized = (series - col_min) / (col_max - col_min)
        variance = float(normalized.var())
        if variance < threshold:
            low_variance[col] = round(variance, 6)
    return low_variance


def _find_highly_correlated_pairs(
    df: pd.DataFrame, numeric_cols: list[str], threshold: float
) -> list[dict[str, Any]]:
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].replace([np.inf, -np.inf], np.nan).corr(numeric_only=True)
    pairs = []
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append(
                    {"feature_a": col_a, "feature_b": col_b, "correlation": round(float(value), 4)}
                )
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return pairs


def generate_profile(
    df: pd.DataFrame,
    *,
    label_column: str = "Label",
    low_variance_threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> DataProfile:
    """Compute the full Milestone 1 data profile for `df`. Read-only."""
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    missing_counts = df.isna().sum()
    missing_values = {
        col: {"count": int(count), "pct": round(float(count) / n_rows * 100, 4)}
        for col, count in missing_counts.items()
        if count > 0
    }

    infinity_values = {}
    for col in numeric_cols:
        inf_count = int(np.isinf(df[col]).sum())
        if inf_count > 0:
            infinity_values[col] = inf_count

    duplicate_row_count = int(df.duplicated().sum())

    class_distribution = None
    if label_column in df.columns:
        counts = df[label_column].value_counts(dropna=False)
        class_distribution = {
            str(label): {"count": int(count), "pct": round(float(count) / n_rows * 100, 4)}
            for label, count in counts.items()
        }

    summary_statistics = df[numeric_cols].describe().to_dict() if numeric_cols else {}

    constant_columns = _find_constant_columns(df)
    low_variance_columns = _find_low_variance_columns(
        df, numeric_cols, exclude=set(constant_columns), threshold=low_variance_threshold
    )
    highly_correlated_pairs = _find_highly_correlated_pairs(
        df, numeric_cols, threshold=correlation_threshold
    )

    return DataProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
        column_dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        missing_values=missing_values,
        infinity_values=infinity_values,
        duplicate_row_count=duplicate_row_count,
        duplicate_row_pct=round(duplicate_row_count / n_rows * 100, 4) if n_rows else 0.0,
        class_distribution=class_distribution,
        summary_statistics=summary_statistics,
        constant_columns=constant_columns,
        low_variance_columns=low_variance_columns,
        highly_correlated_pairs=highly_correlated_pairs,
        label_column=label_column if label_column in df.columns else None,
    )
