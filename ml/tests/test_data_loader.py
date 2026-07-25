import pytest

from ml.data.loader import (
    DatasetNotFoundError,
    load_cicids_csv,
    load_dataset,
    resolve_dataset_path,
)


def test_load_cicids_csv_normalizes_whitespace_only(tmp_path, cicids_like_df):
    csv_path = tmp_path / "sample.csv"
    cicids_like_df.to_csv(csv_path, index=False)

    loaded = load_cicids_csv(csv_path)

    assert list(loaded.columns) == [c.strip() for c in cicids_like_df.columns]
    assert loaded.shape == cicids_like_df.shape
    # Values themselves are untouched.
    assert loaded["Label"].tolist() == cicids_like_df[" Label"].tolist()


def test_load_cicids_csv_can_skip_normalization(tmp_path, cicids_like_df):
    csv_path = tmp_path / "sample.csv"
    cicids_like_df.to_csv(csv_path, index=False)

    loaded = load_cicids_csv(csv_path, normalize_column_names=False)

    assert list(loaded.columns) == list(cicids_like_df.columns)


def test_resolve_dataset_path_raises_when_missing(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        resolve_dataset_path("wednesday", raw_dir=tmp_path)


def test_resolve_dataset_path_rejects_unknown_key(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset key"):
        resolve_dataset_path("not_a_real_day", raw_dir=tmp_path)


def test_load_dataset_resolves_and_loads(tmp_path, cicids_like_df):
    target = tmp_path / "Wednesday-workingHours.pcap_ISCX.csv"
    cicids_like_df.to_csv(target, index=False)

    df = load_dataset("wednesday", raw_dir=tmp_path)

    assert df.shape == cicids_like_df.shape
    assert "Label" in df.columns
