"""Stratified train/validation/test split for CICIDS2017.

Split before any statistical fitting -- `ml.preprocessing.pipeline` fits its
imputer, scaler, and label encoder on `DatasetSplit.train` alone, so nothing
about the validation/test distributions leaks into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_TRAIN_SIZE = 0.7
DEFAULT_VAL_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15

# Splitting a class three ways needs at least 1 row per split; require a
# small margin above that so rounding in the proportional allocation can't
# starve a split of an already-rare class.
MIN_ROWS_PER_CLASS = 3


class InsufficientClassSamplesError(ValueError):
    """Raised when a class has too few rows to stratify into train/val/test."""


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def stratified_split(
    df: pd.DataFrame,
    *,
    label_column: str = "Label",
    train_size: float = DEFAULT_TRAIN_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
) -> DatasetSplit:
    """Split `df` into stratified train/val/test partitions.

    Raises `InsufficientClassSamplesError` up front if any class has too
    few rows to appear in all three splits, rather than letting the second
    internal `train_test_split` call fail with an opaque sklearn error.
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train_size + val_size + test_size must sum to 1.0, got {total}")

    class_counts = df[label_column].value_counts()
    too_small = class_counts[class_counts < MIN_ROWS_PER_CLASS]
    if not too_small.empty:
        raise InsufficientClassSamplesError(
            "The following classes have too few rows to stratify into train/val/test "
            f"(need >= {MIN_ROWS_PER_CLASS} each): {too_small.to_dict()}"
        )

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[label_column], random_state=random_state
    )
    relative_val_size = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[label_column],
        random_state=random_state,
    )

    return DatasetSplit(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )
