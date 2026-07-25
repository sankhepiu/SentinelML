from fastapi import APIRouter

from app.api.v1.routes import health, model, predict, readiness

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(readiness.router)
api_router.include_router(model.router)
api_router.include_router(predict.router)
