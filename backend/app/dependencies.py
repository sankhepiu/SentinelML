"""FastAPI dependency providers."""

from __future__ import annotations

from fastapi import HTTPException, Request

from ml.inference import Predictor


def get_predictor(request: Request) -> Predictor:
    """The loaded `Predictor`, or a 503 if the model failed to load at startup."""
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check GET /api/v1/ready.")
    return predictor
