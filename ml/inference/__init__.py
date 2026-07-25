"""Framework-agnostic inference contract consumed by the backend (see `predictor.py`).

`ml.inference` -- `Predictor` plus its small data contracts (`ModelMetadata`,
`PredictionResult`) -- is the ONLY part of `ml` the backend is allowed to
import. It never reaches into `ml.preprocessing`, `ml.training`, or
`ml.models` directly.
"""

from ml.inference.predictor import ModelMetadata, PredictionResult, Predictor

__all__ = ["ModelMetadata", "PredictionResult", "Predictor"]
