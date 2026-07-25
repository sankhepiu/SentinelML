"""Feature-statistics helpers shared by data profiling (M1) and preprocessing (M2).

Every function here is pure: it reads `df` and returns a description, never
mutating or dropping anything itself. M1's `ml.data.profile` uses these to
report on the raw dataset; M2's `ml.preprocessing.pipeline` uses the same
functions on the training split to decide which columns to actually drop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_LOW_VARIANCE_THRESHOLD = 0.01
DEFAULT_CORRELATION_THRESHOLD = 0.95


def find_constant_columns(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    """Columns with a single distinct value (including an all-NaN column)."""
    columns = list(df.columns) if columns is None else columns
    return [col for col in columns if df[col].nunique(dropna=False) <= 1]


def find_low_variance_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    exclude: set[str] = frozenset(),
    threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
) -> dict[str, float]:
    """Min-max normalized variance below `threshold`, excluding `exclude` (e.g. constants).

    Raw variance is scale-dependent (a column measured in bytes will always
    look "higher variance" than one measured in seconds), so columns are
    normalized to [0, 1] before comparing.
    """
    low_variance = {}
    for col in columns:
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


def find_highly_correlated_pairs(
    df: pd.DataFrame, columns: list[str], *, threshold: float = DEFAULT_CORRELATION_THRESHOLD
) -> list[dict[str, Any]]:
    """Numeric column pairs with |Pearson r| >= threshold, sorted by |r| descending."""
    if len(columns) < 2:
        return []
    corr = df[columns].replace([np.inf, -np.inf], np.nan).corr(numeric_only=True)
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
