"""
FastAPI application factory.

Creates and configures the FastAPI app with lifespan management,
CORS, structured error responses, and all routers.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from facepipe.api.dependencies import get_pipeline
from facepipe.api.routers import enrollment, recognition, identities, health, metrics
from facepipe.config.settings import get_settings
from facepipe.observability.logging import setup_logging, get_logger

logger = get_logger(__name__)

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load models on startup, cleanup on shutdown."""
    global _start_time
    _start_time = time.time()

    settings = get_settings()
    setup_logging(level=settings.log_level, json_output=not settings.debug)

    logger.info("starting_application", version="2.0.0")

    # Pre-load pipeline and models
    pipeline = get_pipeline()
    pipeline.warmup()

    logger.info("application_ready")
    yield

    logger.info("shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Facial Recognition Platform",
        description="Production-grade facial recognition with quality assessment, anti-spoofing, deepfake detection, and active learning.",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error("unhandled_exception", error=str(exc), request_id=request_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc) if settings.debug else "An unexpected error occurred.",
                "request_id": request_id,
            },
        )

    # Register routers
    app.include_router(enrollment.router, prefix="/api/v1", tags=["Enrollment"])
    app.include_router(recognition.router, prefix="/api/v1", tags=["Recognition"])
    app.include_router(identities.router, prefix="/api/v1", tags=["Identities"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])

    return app


def get_uptime() -> float:
    """Return application uptime in seconds."""
    return time.time() - _start_time if _start_time > 0 else 0.0


# Create the app instance for uvicorn
app = create_app()
