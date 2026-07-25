import pytest

from ml.data.loader import DatasetNotFoundError
from ml.data.report import generate_report


def test_generate_report_raises_when_dataset_missing(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        generate_report("wednesday", raw_dir=tmp_path, reports_dir=tmp_path / "reports")


def test_generate_report_end_to_end(tmp_path, cicids_like_df):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cicids_like_df.to_csv(raw_dir / "Wednesday-workingHours.pcap_ISCX.csv", index=False)
    reports_dir = tmp_path / "reports"

    outputs = generate_report(
        "wednesday", raw_dir=raw_dir, reports_dir=reports_dir, label_column="Label"
    )

    for path in outputs.values():
        assert path.exists()

    report_text = outputs["report"].read_text()
    assert "# CICIDS2017 Data Profile Report" in report_text
    assert "Observed Data Quality Issues" in report_text
    assert "Expected Preprocessing Steps" in report_text
    assert "Potential Modelling Challenges" in report_text
    assert "Fwd URG Flags" in report_text  # constant column called out
    assert "Class imbalance" in report_text

    figures_dir = reports_dir / "figures"
    assert (figures_dir / "class_distribution.png").exists()
    assert (figures_dir / "missing_value_matrix.png").exists()
    assert (figures_dir / "correlation_heatmap.png").exists()
    assert (figures_dir / "top_numerical_feature_distributions.png").exists()

    assert (reports_dir / "data_profile.json").exists()
    assert (reports_dir / "summary_statistics.csv").exists()
    assert (reports_dir / "correlation_pairs.csv").exists()


def test_generate_report_does_not_mutate_source_csv(tmp_path, cicids_like_df):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_csv = raw_dir / "Wednesday-workingHours.pcap_ISCX.csv"
    cicids_like_df.to_csv(source_csv, index=False)
    original_bytes = source_csv.read_bytes()

    generate_report(
        "wednesday", raw_dir=raw_dir, reports_dir=tmp_path / "reports", label_column="Label"
    )

    assert source_csv.read_bytes() == original_bytes
