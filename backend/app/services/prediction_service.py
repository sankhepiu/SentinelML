"""Translates prediction API payloads to/from `ml.inference.Predictor` calls.

Pydantic validates payload *shape* (a mapping of str -> float); this module
validates payload *content* against the currently loaded model's actual
feature set, which isn't known until the model is loaded at startup, so it
can't be expressed as a static Pydantic schema.
"""

from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from app.schemas.prediction import PredictionResponse
from ml.inference import PredictionResult, Predictor


def _validate_feature_names(features: dict[str, float], expected: list[str]) -> None:
    expected_set = set(expected)
    actual_set = set(features)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        detail: dict[str, list[str]] = {}
        if missing:
            detail["missing_features"] = missing
        if unexpected:
            detail["unexpected_features"] = unexpected
        raise HTTPException(status_code=422, detail=detail)


def _to_response(result: PredictionResult, model_version: str) -> PredictionResponse:
    return PredictionResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        class_probabilities=result.class_probabilities,
        model_version=model_version,
    )


def predict_one(predictor: Predictor, features: dict[str, float]) -> PredictionResponse:
    _validate_feature_names(features, predictor.feature_names)
    df = pd.DataFrame([features])
    result = predictor.predict(df)[0]
    return _to_response(result, predictor.metadata.version)


def predict_batch(
    predictor: Predictor, instances: list[dict[str, float]]
) -> list[PredictionResponse]:
    for features in instances:
        _validate_feature_names(features, predictor.feature_names)
    df = pd.DataFrame(instances)
    results = predictor.predict(df)
    return [_to_response(result, predictor.metadata.version) for result in results]
