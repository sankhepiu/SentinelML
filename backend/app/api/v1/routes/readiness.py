from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.readiness import ReadinessResponse

router = APIRouter(tags=["health"])


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "Model artifacts are not loaded"}},
    summary="Readiness probe",
)
def readiness(request: Request) -> JSONResponse:
    """200 once the model and preprocessing pipeline are loaded; 503 otherwise.

    Distinct from `/health`: `/health` only says the process is alive,
    `/ready` says it can actually serve predictions -- the two diverge
    whenever model loading fails at startup (see `predictor_error`).
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        error = getattr(request.app.state, "predictor_error", None) or "model not loaded"
        body = ReadinessResponse(ready=False, detail=error)
        return JSONResponse(status_code=503, content=body.model_dump())
    return JSONResponse(status_code=200, content=ReadinessResponse(ready=True).model_dump())
