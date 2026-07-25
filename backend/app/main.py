"""FastAPI application factory.

`app = create_app()` below is the real ASGI entrypoint (`app.main:app`, what
`uvicorn`/`sentinel serve` runs). `create_app()` also takes an explicit
`Settings` so tests can build an isolated app instance pointed at a
different `model_registry_path` without touching process-wide state.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.middleware import RequestLoggingMiddleware
from ml.inference import Predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    try:
        predictor = Predictor.from_registry(
            settings.resolved_model_registry_path, version=settings.model_version
        )
        app.state.predictor = predictor
        app.state.predictor_error = None
        logger.info(
            "model_loaded",
            extra={
                "model_type": predictor.metadata.model_type,
                "model_version": predictor.metadata.version,
            },
        )
    except Exception as exc:
        app.state.predictor = None
        app.state.predictor_error = str(exc)
        logger.error("model_load_failed", extra={"error": str(exc)})
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        description=(
            "SentinelML network intrusion detection inference API. Serves predictions "
            "from the best model selected in Milestone 3 training."
        ),
        lifespan=_lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": "An unexpected error occurred."},
        )

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
