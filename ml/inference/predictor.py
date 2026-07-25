"""The contract boundary between the ML pipeline and the API layer.

`Predictor` loads a versioned model artifact (produced by any trainer in
`ml.training`) plus the fitted `ml.preprocessing.pipeline.PreprocessingPipeline`
it was trained against, and exposes one prediction surface regardless of the
underlying framework. The backend depends only on `ml.inference` (this
module), never on scikit-learn/XGBoost/`ml.preprocessing`/`ml.models`
directly, so swapping or adding model types in `ml/` never requires a
change in `backend/app`. Preprocessing is always loaded, never refit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.preprocessing.pipeline import PreprocessingPipeline


@dataclass
class ModelMetadata:
    model_type: str
    version: str
    feature_names: list[str]
    metrics: dict[str, float]


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float] | None


class Predictor:
    def __init__(self, model_dir: Path, *, preprocessing_dir: Path | None = None):
        self.model_dir = Path(model_dir)
        # By convention (see ml/models/registry.py), preprocessing artifacts live in a
        # directory named "preprocessing" alongside the versioned model directories.
        self.preprocessing_dir = (
            Path(preprocessing_dir)
            if preprocessing_dir
            else self.model_dir.parent / "preprocessing"
        )
        self.metadata: ModelMetadata | None = None
        self._pipeline: PreprocessingPipeline | None = None
        self._model: Any = None

    @classmethod
    def from_registry(cls, models_root: str | Path, *, version: str | None = None) -> Predictor:
        """Resolve `version` (or the latest) via `ml.models.registry.ModelRegistry` and load it."""
        from ml.models.registry import ModelRegistry

        model_dir = ModelRegistry(Path(models_root)).resolve(version)
        return cls(model_dir).load()

    def load(self) -> Predictor:
        """Load the fitted preprocessing pipeline and trained model. Never fits either."""
        self._pipeline = PreprocessingPipeline.load(self.preprocessing_dir)
        self._model = joblib.load(self.model_dir / "model.joblib")
        metadata_dict = json.loads((self.model_dir / "metadata.json").read_text())
        self.metadata = ModelMetadata(**metadata_dict)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._pipeline is not None

    @property
    def feature_names(self) -> list[str]:
        """Exact feature names (and order) the model expects, from the fitted pipeline."""
        self._check_loaded()
        return self._pipeline.metadata.feature_columns

    @property
    def label_classes(self) -> list[str]:
        """Every class name the model can predict, in label-encoded order."""
        self._check_loaded()
        return self._pipeline.metadata.label_classes

    @property
    def training_summary(self) -> dict[str, Any] | None:
        """The full training comparison report (`ml.training.run.TrainingRunSummary`), if present.

        Written by `sentinel train` alongside `model.joblib`/`metadata.json`.
        Returns `None` rather than raising if a model directory was
        populated some other way and lacks it -- this is presentation data
        for a dashboard, not required for prediction.
        """
        self._check_loaded()
        path = self.model_dir / "training_summary.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def predict(self, df: pd.DataFrame) -> list[PredictionResult]:
        """Preprocess `df` (raw feature columns, no label) and predict one result per row."""
        self._check_loaded()
        X = self._pipeline.transform_features(df)
        proba = np.asarray(self._model.predict_proba(X))
        classes = np.asarray(self._model.classes_)
        class_names = self._pipeline.decode_labels(classes)

        best_idx = np.argmax(proba, axis=1)
        results = []
        for row, idx in enumerate(best_idx):
            class_probabilities = {
                str(name): float(p) for name, p in zip(class_names, proba[row], strict=True)
            }
            results.append(
                PredictionResult(
                    predicted_class=str(class_names[idx]),
                    confidence=float(proba[row, idx]),
                    class_probabilities=class_probabilities,
                )
            )
        return results

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Raw per-class probabilities (columns follow `self._model.classes_` order)."""
        self._check_loaded()
        X = self._pipeline.transform_features(df)
        return np.asarray(self._model.predict_proba(X))

    def _check_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Predictor must be load()ed before use.")
