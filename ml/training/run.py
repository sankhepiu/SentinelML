"""Milestone 3 training pipeline: load processed splits + preprocessing
artifacts -> train candidate models -> evaluate -> select best -> persist.

Consumes Milestone 2's output directly: `ml/data/processed/{train,val,test}.csv`
(already cleaned, imputed, scaled, and label-encoded) and the fitted
`ml.preprocessing.pipeline.PreprocessingPipeline` artifacts. Preprocessing
is loaded for its metadata (feature names, label mapping) only -- it is
never refit here.

Run via the CLI:

    uv run sentinel train
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from ml.evaluation.metrics import EvaluationResult, evaluate_predictions
from ml.evaluation.visualize import plot_confusion_matrix, plot_feature_importance
from ml.models.registry import ModelRegistry
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.preprocessing.run import DEFAULT_ARTIFACTS_DIR as DEFAULT_PREPROCESSING_ARTIFACTS_DIR
from ml.preprocessing.run import DEFAULT_PROCESSED_DIR
from ml.training.base import BaseModelTrainer
from ml.training.random_forest import RandomForestTrainer
from ml.training.xgboost_trainer import XGBoostTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "ml" / "models" / "artifacts"

DEFAULT_SELECTION_METRIC = "f1_macro"
DEFAULT_N_ESTIMATORS = 200
DEFAULT_RANDOM_STATE = 42


@dataclass
class TrainingRunSummary:
    version: str
    best_model_type: str
    selection_metric: str
    trained_models: list[str]
    skipped_models: dict[str, str]
    val_metrics: dict[str, dict]
    test_metrics: dict
    feature_importances: dict[str, dict[str, float]]
    class_distribution: dict[str, dict[str, int]]
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    feature_columns: list[str]
    label_mapping: dict[str, str]
    random_state: int


def _class_distribution(y: np.ndarray, label_mapping: dict[str, str]) -> dict[str, int]:
    counts = pd.Series(y).value_counts()
    return {label_mapping[str(code)]: int(count) for code, count in counts.items()}


def get_available_trainer_classes() -> tuple[dict[str, type[BaseModelTrainer]], dict[str, str]]:
    """Random Forest and XGBoost are always available; LightGBM only if importable.

    Returns (available trainer classes by model_type, {skipped model_type: reason}).
    """
    available: dict[str, type[BaseModelTrainer]] = {
        "random_forest": RandomForestTrainer,
        "xgboost": XGBoostTrainer,
    }
    skipped: dict[str, str] = {}
    try:
        from ml.training.lightgbm_trainer import LightGBMTrainer

        available["lightgbm"] = LightGBMTrainer
    except Exception as exc:  # pragma: no cover - platform dependent (e.g. missing libomp)
        skipped["lightgbm"] = str(exc)
    return available, skipped


def run_training(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    preprocessing_artifacts_dir: str | Path = DEFAULT_PREPROCESSING_ARTIFACTS_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    label_column: str = "Label",
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    preprocessing_artifacts_dir = Path(preprocessing_artifacts_dir)
    models_dir = Path(models_dir)

    # Loaded for metadata only (feature names, label mapping) -- never refit.
    preprocessing = PreprocessingPipeline.load(preprocessing_artifacts_dir)
    metadata = preprocessing.metadata
    feature_columns = metadata.feature_columns
    label_mapping = metadata.label_mapping
    class_labels = sorted(int(code) for code in label_mapping)
    target_names = [label_mapping[str(code)] for code in class_labels]

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    missing = set(feature_columns) - set(train_df.columns)
    if missing:
        raise ValueError(
            f"Processed data at {processed_dir} is missing columns the fitted preprocessing "
            f"artifacts at {preprocessing_artifacts_dir} expect: {sorted(missing)}"
        )

    X_train, y_train = train_df[feature_columns], train_df[label_column].to_numpy()
    X_val, y_val = val_df[feature_columns], val_df[label_column].to_numpy()
    X_test, y_test = test_df[feature_columns], test_df[label_column].to_numpy()

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    trainer_classes, skipped_models = get_available_trainer_classes()

    val_results: dict[str, EvaluationResult] = {}
    fitted_trainers: dict[str, BaseModelTrainer] = {}
    for model_type, trainer_cls in trainer_classes.items():
        trainer = trainer_cls(n_estimators=n_estimators, random_state=random_state)
        trainer.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = trainer.predict(X_val)
        y_proba = trainer.predict_proba(X_val)
        val_results[model_type] = evaluate_predictions(
            y_val, y_pred, y_proba, labels=class_labels, target_names=target_names
        )
        fitted_trainers[model_type] = trainer

    best_model_type = max(val_results, key=lambda m: getattr(val_results[m], selection_metric))
    best_trainer = fitted_trainers[best_model_type]

    # Final, unbiased evaluation of the selected model on the held-out test split.
    y_test_pred = best_trainer.predict(X_test)
    y_test_proba = best_trainer.predict_proba(X_test)
    test_result = evaluate_predictions(
        y_test, y_test_pred, y_test_proba, labels=class_labels, target_names=target_names
    )

    registry = ModelRegistry(models_dir)
    version = registry.next_version()
    version_dir = models_dir / version
    figures_dir = version_dir / "figures"

    outputs: dict[str, Path] = {"model": best_trainer.save(version_dir)}

    feature_importances: dict[str, dict[str, float]] = {}
    for model_type, result in val_results.items():
        importances = fitted_trainers[model_type].feature_importances()
        feature_importances[model_type] = {
            col: float(value) for col, value in zip(feature_columns, importances, strict=True)
        }

        outputs[f"figure_confusion_matrix_{model_type}"] = plot_confusion_matrix(
            result.confusion_matrix,
            target_names,
            figures_dir / f"confusion_matrix_{model_type}.png",
            title=f"{model_type} -- validation confusion matrix",
        )
        outputs[f"figure_feature_importance_{model_type}"] = plot_feature_importance(
            feature_columns,
            importances,
            figures_dir / f"feature_importance_{model_type}.png",
            title=f"{model_type} -- feature importance",
        )

    model_metrics = {
        "accuracy": test_result.accuracy,
        "precision_macro": test_result.precision_macro,
        "recall_macro": test_result.recall_macro,
        "f1_macro": test_result.f1_macro,
        "f1_weighted": test_result.f1_weighted,
    }
    if test_result.roc_auc_ovr_macro is not None:
        model_metrics["roc_auc_ovr_macro"] = test_result.roc_auc_ovr_macro

    model_metadata = {
        "model_type": best_model_type,
        "version": version,
        "feature_names": feature_columns,
        "metrics": model_metrics,
    }
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(model_metadata, indent=2))
    outputs["metadata"] = metadata_path

    class_distribution = {
        "train": _class_distribution(y_train, label_mapping),
        "val": _class_distribution(y_val, label_mapping),
        "test": _class_distribution(y_test, label_mapping),
    }

    summary = TrainingRunSummary(
        version=version,
        best_model_type=best_model_type,
        selection_metric=selection_metric,
        trained_models=list(val_results),
        skipped_models=skipped_models,
        val_metrics={m: r.to_dict() for m, r in val_results.items()},
        test_metrics=test_result.to_dict(),
        feature_importances=feature_importances,
        class_distribution=class_distribution,
        n_train_rows=len(train_df),
        n_val_rows=len(val_df),
        n_test_rows=len(test_df),
        feature_columns=feature_columns,
        label_mapping=label_mapping,
        random_state=random_state,
    )
    summary_path = version_dir / "training_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))
    outputs["training_summary"] = summary_path

    return outputs
