"""LightGBM baseline trainer.

Imported lazily by `ml.training.run` -- LightGBM's compiled backend isn't
guaranteed to load on every platform (e.g. it needs libomp on macOS), so
callers should catch import failures and treat it as unavailable rather
than importing this module unconditionally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ml.training.base import BaseModelTrainer


class LightGBMTrainer(BaseModelTrainer):
    model_type = "lightgbm"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(
        self, x: pd.DataFrame, y: pd.Series, *, sample_weight: np.ndarray | None = None
    ) -> LightGBMTrainer:
        self.model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x: pd.DataFrame) -> Any:
        return self.model.predict(x)

    def predict_proba(self, x: pd.DataFrame) -> Any:
        return self.model.predict_proba(x)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "model.joblib"
        joblib.dump(self.model, path)
        return path
