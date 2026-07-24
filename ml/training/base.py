"""Trainer contract shared by every model implementation.

Random Forest and XGBoost (and future additions such as LightGBM, CatBoost,
or a neural network) each get their own module here, implementing this same
interface. Nothing outside `ml/` depends on a concrete trainer class -- the
backend only ever consumes artifacts through `ml.inference.Predictor`, so
adding a new model type never requires touching the API layer.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseModelTrainer(ABC):
    model_type: str

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BaseModelTrainer":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: Path) -> Path:
        raise NotImplementedError
