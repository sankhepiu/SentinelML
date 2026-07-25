import numpy as np
import pandas as pd

from ml.preprocessing.cleaning import drop_duplicate_rows, replace_infinite_with_nan


def test_drop_duplicate_rows_removes_seeded_duplicates(cicids_like_df):
    before = len(cicids_like_df)

    deduped = drop_duplicate_rows(cicids_like_df)

    assert len(deduped) < before
    assert deduped.duplicated().sum() == 0


def test_drop_duplicate_rows_keeps_first_occurrence():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})

    deduped = drop_duplicate_rows(df)

    assert len(deduped) == 2
    assert list(deduped["a"]) == [1, 2]


def test_drop_duplicate_rows_resets_index():
    df = pd.DataFrame({"a": [1, 1, 2]}, index=[5, 6, 7])

    deduped = drop_duplicate_rows(df)

    assert list(deduped.index) == list(range(len(deduped)))


def test_replace_infinite_with_nan_converts_both_signs():
    df = pd.DataFrame({"x": [1.0, np.inf, -np.inf, 4.0]})

    cleaned = replace_infinite_with_nan(df)

    assert cleaned["x"].isna().sum() == 2
    assert np.isinf(cleaned["x"]).sum() == 0


def test_replace_infinite_with_nan_does_not_mutate_input():
    df = pd.DataFrame({"x": [1.0, np.inf]})
    before = df.copy()

    replace_infinite_with_nan(df)

    pd.testing.assert_frame_equal(df, before)


def test_replace_infinite_with_nan_respects_column_subset():
    df = pd.DataFrame({"x": [np.inf], "y": [np.inf]})

    cleaned = replace_infinite_with_nan(df, columns=["x"])

    assert cleaned["x"].isna().all()
    assert np.isinf(cleaned["y"]).all()
