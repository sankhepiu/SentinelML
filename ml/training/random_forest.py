"""Random Forest baseline trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.training.base import BaseModelTrainer


class RandomForestTrainer(BaseModelTrainer):
    model_type = "random_forest"

    def __init__(
        self, *, n_estimators: int = 200, max_depth: int | None = None, random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(
        self, x: pd.DataFrame, y: pd.Series, *, sample_weight: np.ndarray | None = None
    ) -> RandomForestTrainer:
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
