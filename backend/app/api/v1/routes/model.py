from fastapi import APIRouter, Depends

from app.dependencies import get_predictor
from app.schemas.model import ModelInfoResponse
from ml.inference import Predictor

router = APIRouter(tags=["model"])


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    responses={503: {"description": "Model is not loaded -- check GET /ready"}},
    summary="Metadata about the currently loaded model",
)
def model_info(predictor: Predictor = Depends(get_predictor)) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_type=predictor.metadata.model_type,
        model_version=predictor.metadata.version,
        feature_names=predictor.feature_names,
        label_classes=predictor.label_classes,
        metrics=predictor.metadata.metrics,
    )
