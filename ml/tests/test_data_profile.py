import json

from ml.data.profile import generate_profile


def test_generate_profile_dimensions_and_memory(cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column=" Label")

    assert profile.n_rows == len(cicids_like_df)
    assert profile.n_cols == cicids_like_df.shape[1]
    assert profile.memory_usage_bytes > 0


def test_generate_profile_does_not_mutate_input(cicids_like_df):
    before = cicids_like_df.copy(deep=True)

    generate_profile(cicids_like_df, label_column=" Label")

    assert cicids_like_df.equals(before)


def test_generate_profile_detects_missing_values(cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column=" Label")

    assert " Fwd Packet Length Max" in profile.missing_values
    assert " Total Backward Packets" in profile.missing_values
    assert profile.missing_values[" Fwd Packet Length Max"]["count"] > 0


def test_generate_profile_detects_infinity_values(cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column=" Label")

    assert "Flow Bytes/s" in profile.infinity_values
    assert " Flow Packets/s" in profile.infinity_values
    assert profile.infinity_values["Flow Bytes/s"] >= 10


def test_generate_profile_detects_duplicate_rows(cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column=" Label")

    assert profile.duplicate_row_count >= 5


def test_generate_profile_detects_constant_columns(cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column=" Label")

    assert " Fwd URG Flags" in profile.constant_columns


def test_generate_profile_detects_low_variance_columns(cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column=" Label")

    assert " Bwd PSH Flags" in profile.low_variance_columns
    assert " Fwd URG Flags" not in profile.low_variance_columns  # constant, not low-variance


def test_generate_profile_detects_highly_correlated_pairs(cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column=" Label")

    pairs = {frozenset((p["feature_a"], p["feature_b"])) for p in profile.highly_correlated_pairs}
    assert frozenset((" Total Fwd Packets", "Subflow Fwd Packets")) in pairs


def test_generate_profile_class_distribution(cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column=" Label")

    assert profile.class_distribution is not None
    assert set(profile.class_distribution) == {"BENIGN", "DDoS", "PortScan"}
    assert (
        profile.class_distribution["BENIGN"]["count"] > profile.class_distribution["DDoS"]["count"]
    )


def test_generate_profile_missing_label_column_is_none(cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column="NotARealColumn")

    assert profile.class_distribution is None
    assert profile.label_column is None


def test_data_profile_to_json_is_serializable(tmp_path, cicids_like_df):
    profile = generate_profile(cicids_like_df, label_column=" Label")

    path = profile.to_json(tmp_path / "profile.json")

    data = json.loads(path.read_text())
    assert data["n_rows"] == len(cicids_like_df)
    assert isinstance(data["highly_correlated_pairs"], list)
