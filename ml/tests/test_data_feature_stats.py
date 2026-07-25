import numpy as np

from ml.data.feature_stats import (
    find_constant_columns,
    find_highly_correlated_pairs,
    find_low_variance_columns,
)


def test_find_constant_columns_detects_single_value_column(cicids_like_df):
    constants = find_constant_columns(cicids_like_df)

    assert " Fwd URG Flags" in constants


def test_find_constant_columns_respects_explicit_column_subset(cicids_like_df):
    constants = find_constant_columns(cicids_like_df, columns=[" Destination Port"])

    assert constants == []


def test_find_constant_columns_treats_all_nan_column_as_constant():
    import pandas as pd

    df = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan]})

    assert find_constant_columns(df) == ["all_nan"]


def test_find_low_variance_columns_detects_mostly_constant_column(cicids_like_df):
    numeric_cols = cicids_like_df.select_dtypes(include=[np.number]).columns.tolist()

    low_variance = find_low_variance_columns(cicids_like_df, numeric_cols)

    assert " Bwd PSH Flags" in low_variance
    assert low_variance[" Bwd PSH Flags"] < 0.01


def test_find_low_variance_columns_excludes_given_columns(cicids_like_df):
    numeric_cols = cicids_like_df.select_dtypes(include=[np.number]).columns.tolist()

    low_variance = find_low_variance_columns(
        cicids_like_df, numeric_cols, exclude={" Bwd PSH Flags"}
    )

    assert " Bwd PSH Flags" not in low_variance


def test_find_highly_correlated_pairs_detects_seeded_pair(cicids_like_df):
    numeric_cols = cicids_like_df.select_dtypes(include=[np.number]).columns.tolist()

    pairs = find_highly_correlated_pairs(cicids_like_df, numeric_cols)

    matched = {frozenset((p["feature_a"], p["feature_b"])) for p in pairs}
    assert frozenset((" Total Fwd Packets", "Subflow Fwd Packets")) in matched


def test_find_highly_correlated_pairs_returns_empty_for_single_column(cicids_like_df):
    assert find_highly_correlated_pairs(cicids_like_df, [" Destination Port"]) == []
