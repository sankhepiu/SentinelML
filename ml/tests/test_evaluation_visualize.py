import numpy as np

from ml.evaluation.visualize import plot_confusion_matrix, plot_feature_importance


def test_plot_confusion_matrix_creates_file(tmp_path):
    cm = [[10, 2], [1, 15]]

    path = plot_confusion_matrix(cm, ["benign", "attack"], tmp_path / "cm.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_feature_importance_creates_file(tmp_path):
    names = [f"f{i}" for i in range(10)]
    importances = np.linspace(0.01, 0.5, num=10)

    path = plot_feature_importance(names, importances, tmp_path / "fi.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_feature_importance_respects_top_n(tmp_path, monkeypatch):
    captured = {}
    import ml.evaluation.visualize as viz

    original_barplot = viz.sns.barplot

    def spy_barplot(*args, **kwargs):
        captured["y"] = list(kwargs["y"])
        return original_barplot(*args, **kwargs)

    monkeypatch.setattr(viz.sns, "barplot", spy_barplot)

    names = [f"f{i}" for i in range(20)]
    importances = np.arange(20, dtype=float)  # f19 is most important, f0 least

    plot_feature_importance(names, importances, tmp_path / "fi.png", top_n=5)

    assert len(captured["y"]) == 5
    assert captured["y"][0] == "f19"
