from fastapi import APIRouter, Depends

from app.dependencies import get_predictor
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import predict_batch, predict_one
from ml.inference import Predictor

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"description": "Feature payload doesn't match the model's expected feature set"},
        503: {"description": "Model is not loaded -- check GET /ready"},
    },
    summary="Predict the class of a single network flow",
)
def predict(
    payload: PredictionRequest, predictor: Predictor = Depends(get_predictor)
) -> PredictionResponse:
    return predict_one(predictor, payload.features)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        422: {"description": "One or more feature payloads don't match the model's feature set"},
        503: {"description": "Model is not loaded -- check GET /ready"},
    },
    summary="Predict the class of a batch of network flows",
)
def predict_batch_endpoint(
    payload: BatchPredictionRequest, predictor: Predictor = Depends(get_predictor)
) -> BatchPredictionResponse:
    predictions = predict_batch(predictor, payload.instances)
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
