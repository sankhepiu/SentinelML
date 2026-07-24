"""The contract boundary between the ML pipeline and the API layer.

`Predictor` loads a versioned model artifact (produced by any trainer in
`ml.training`) and exposes one prediction surface regardless of the
underlying framework. The backend depends only on this class, never on
scikit-learn/XGBoost/etc. directly, so swapping or adding model types in
`ml/` never requires a change in `backend/app`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ModelMetadata:
    model_type: str
    version: str
    feature_names: list[str]
    metrics: dict[str, float]


class Predictor:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.metadata: ModelMetadata | None = None
        self._pipeline: Any = None

    def load(self) -> "Predictor":
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> Any:
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> Any:
        raise NotImplementedError
