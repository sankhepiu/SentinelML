"""Classification evaluation metrics.

`evaluate_predictions` computes every metric Milestone 3 requires: accuracy,
precision/recall/F1 (macro and weighted), ROC-AUC (one-vs-rest, when
probability estimates are available), a confusion matrix, and a full
per-class classification report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    roc_auc_ovr_macro: float | None
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    *,
    labels: list[int],
    target_names: list[str],
) -> EvaluationResult:
    """Evaluate predictions against `y_true`.

    `labels` must be the full set of class codes the model was fit on (in
    the order `y_proba`'s columns follow), so ROC-AUC and the confusion
    matrix line up even if a class is entirely absent from this split.
    """
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
            )
        except ValueError:
            # e.g. a class in `labels` never appears in y_true for this split.
            roc_auc = None

    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        precision_weighted=float(
            precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        recall_macro=float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        recall_weighted=float(
            recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        f1_macro=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        f1_weighted=float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        roc_auc_ovr_macro=roc_auc,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        classification_report=classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    )
