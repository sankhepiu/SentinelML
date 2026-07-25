from ml.data.visualize import (
    generate_all_visualizations,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_missing_value_matrix,
    plot_top_numeric_distributions,
)


def test_plot_class_distribution_creates_file(tmp_path, cicids_like_df):
    path = plot_class_distribution(cicids_like_df, " Label", tmp_path / "class.png")

    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_class_distribution_returns_none_without_label(tmp_path, cicids_like_df):
    path = plot_class_distribution(cicids_like_df, "NotARealColumn", tmp_path / "class.png")

    assert path is None


def test_plot_missing_value_matrix_creates_file(tmp_path, cicids_like_df):
    path = plot_missing_value_matrix(cicids_like_df, tmp_path / "missing.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_correlation_heatmap_creates_file(tmp_path, cicids_like_df):
    path = plot_correlation_heatmap(cicids_like_df, tmp_path / "corr.png")

    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_top_numeric_distributions_creates_file(tmp_path, cicids_like_df):
    path = plot_top_numeric_distributions(cicids_like_df, tmp_path / "top.png")

    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_generate_all_visualizations_writes_every_figure(tmp_path, cicids_like_df):
    figures = generate_all_visualizations(cicids_like_df, tmp_path, label_column=" Label")

    assert set(figures) == {
        "class_distribution",
        "missing_value_matrix",
        "correlation_heatmap",
        "top_numerical_feature_distributions",
    }
    for path in figures.values():
        assert path is not None
        assert path.exists()
