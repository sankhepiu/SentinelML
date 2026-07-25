from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_predictor
from app.schemas.model import ModelInfoResponse
from app.schemas.training_summary import TrainingSummaryResponse
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


@router.get(
    "/model/training-summary",
    response_model=TrainingSummaryResponse,
    responses={
        404: {"description": "This model version has no training summary on disk"},
        503: {"description": "Model is not loaded -- check GET /ready"},
    },
    summary="Full training comparison report for the loaded model",
)
def model_training_summary(
    predictor: Predictor = Depends(get_predictor),
) -> TrainingSummaryResponse:
    summary = predictor.training_summary
    if summary is None:
        raise HTTPException(
            status_code=404, detail="No training summary available for this model version."
        )
    return TrainingSummaryResponse(**summary)
