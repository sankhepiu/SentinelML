"""Deterministic, dataset-wide cleaning steps that run before splitting.

Both steps here are structural rather than statistical -- they don't fit
anything, so applying them to the full dataset before the train/val/test
split doesn't leak label information. Deduplication in particular *must*
happen before splitting: dropping duplicates independently per split would
let identical rows land in both train and test, which is a worse leak than
the one splitting-before-cleaning would avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate rows, keeping the first occurrence."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def replace_infinite_with_nan(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace +/-Inf with NaN so downstream imputers can handle it uniformly.

    Only numeric columns are affected; `columns` restricts which ones.
    """
    df = df.copy()
    target_cols = columns if columns is not None else df.select_dtypes(include=[np.number]).columns
    df[target_cols] = df[target_cols].replace([np.inf, -np.inf], np.nan)
    return df
