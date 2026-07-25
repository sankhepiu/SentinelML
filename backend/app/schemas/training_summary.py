"""Response schema for GET /model/training-summary.

Mirrors `ml.training.run.TrainingRunSummary` (written by `sentinel train`
alongside the model artifact) -- the full comparison across every
candidate model trained in that run, not just the one that was selected.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationMetrics(BaseModel):
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    roc_auc_ovr_macro: float | None = Field(
        None, description="One-vs-rest macro ROC-AUC, when probability estimates were available"
    )
    confusion_matrix: list[list[int]] = Field(
        ..., description="Rows = true class, columns = predicted class, in label-encoded order"
    )
    classification_report: dict[str, Any] = Field(
        ...,
        description="sklearn classification_report(output_dict=True): per-class precision/recall/F1/support",
    )


class TrainingSummaryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    version: str
    best_model_type: str = Field(..., description="Which candidate won model selection")
    selection_metric: str = Field(..., description="Validation metric used to pick the winner")
    trained_models: list[str]
    skipped_models: dict[str, str] = Field(
        ..., description="Candidates that couldn't be trained in this environment, and why"
    )
    val_metrics: dict[str, EvaluationMetrics] = Field(
        ..., description="Every trained candidate's validation-split evaluation"
    )
    test_metrics: EvaluationMetrics = Field(
        ..., description="The selected model's held-out test-split evaluation"
    )
    feature_importances: dict[str, dict[str, float]] = Field(
        ..., description="Feature name -> importance, per trained candidate"
    )
    class_distribution: dict[str, dict[str, int]] = Field(
        ..., description="Per-class row counts in each of the train/val/test splits"
    )
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    feature_columns: list[str]
    label_mapping: dict[str, str]
    random_state: int
