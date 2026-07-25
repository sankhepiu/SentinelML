"""Evaluation visualizations: confusion matrices and feature importance, saved as PNGs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

TOP_N_FEATURES = 20


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    confusion_matrix: list[list[int]] | np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    *,
    title: str = "Confusion Matrix",
) -> Path:
    cm = np.asarray(confusion_matrix)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.9), max(5, len(class_names) * 0.8)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Row count"},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, output_path)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_path: str | Path,
    *,
    top_n: int = TOP_N_FEATURES,
    title: str = "Feature Importance",
) -> Path:
    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in order]
    top_importances = np.asarray(importances)[order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(top_features) * 0.35)))
    sns.barplot(
        x=top_importances, y=top_features, hue=top_features, palette="viridis", legend=False, ax=ax
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    return _save(fig, output_path)
