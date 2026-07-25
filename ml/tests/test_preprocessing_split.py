import pandas as pd
import pytest

from ml.preprocessing.split import InsufficientClassSamplesError, stratified_split


def test_stratified_split_produces_expected_sizes(cicids_like_df):
    split = stratified_split(cicids_like_df, label_column=" Label", random_state=42)

    total = len(cicids_like_df)
    assert len(split.train) + len(split.val) + len(split.test) == total
    assert abs(len(split.train) / total - 0.7) < 0.03
    assert abs(len(split.val) / total - 0.15) < 0.03
    assert abs(len(split.test) / total - 0.15) < 0.03


def test_stratified_split_preserves_class_proportions(cicids_like_df):
    split = stratified_split(cicids_like_df, label_column=" Label", random_state=42)

    full_props = cicids_like_df[" Label"].value_counts(normalize=True)
    train_props = split.train[" Label"].value_counts(normalize=True)

    for label, prop in full_props.items():
        assert abs(train_props.get(label, 0) - prop) < 0.05


def test_stratified_split_every_class_present_in_every_split(cicids_like_df):
    split = stratified_split(cicids_like_df, label_column=" Label", random_state=42)

    classes = set(cicids_like_df[" Label"].unique())
    assert set(split.train[" Label"].unique()) == classes
    assert set(split.val[" Label"].unique()) == classes
    assert set(split.test[" Label"].unique()) == classes


def test_stratified_split_no_row_overlap_between_splits(cicids_like_df):
    df = cicids_like_df.reset_index(drop=True).assign(_row_id=range(len(cicids_like_df)))

    split = stratified_split(df, label_column=" Label", random_state=42)

    train_ids = set(split.train["_row_id"])
    val_ids = set(split.val["_row_id"])
    test_ids = set(split.test["_row_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_stratified_split_is_deterministic_given_seed(cicids_like_df):
    split_a = stratified_split(cicids_like_df, label_column=" Label", random_state=7)
    split_b = stratified_split(cicids_like_df, label_column=" Label", random_state=7)

    pd.testing.assert_frame_equal(split_a.train, split_b.train)


def test_stratified_split_rejects_ratios_not_summing_to_one(cicids_like_df):
    with pytest.raises(ValueError, match="must sum to 1.0"):
        stratified_split(
            cicids_like_df,
            label_column=" Label",
            train_size=0.5,
            val_size=0.2,
            test_size=0.2,
        )


def test_stratified_split_raises_for_rare_class():
    df = pd.DataFrame(
        {
            "feature": range(20),
            "Label": ["BENIGN"] * 18 + ["RareAttack"] * 2,
        }
    )

    with pytest.raises(InsufficientClassSamplesError, match="RareAttack"):
        stratified_split(df, label_column="Label")
