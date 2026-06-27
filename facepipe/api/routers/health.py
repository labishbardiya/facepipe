"""
Health check API router.

GET /api/v1/health — Service health
GET /api/v1/health/ready — Readiness probe
GET /api/v1/health/live — Liveness probe
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from facepipe.api.dependencies import get_pipeline, get_identity_manager
from facepipe.api.schemas import HealthResponse, ReadinessResponse
from facepipe.core.pipeline import RecognitionPipeline
from facepipe.storage.identity_manager import IdentityManager

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    pipeline: RecognitionPipeline = Depends(get_pipeline),
    identity_mgr: IdentityManager = Depends(get_identity_manager),
) -> HealthResponse:
    """Service health check."""
    from facepipe.api.app import get_uptime

    return HealthResponse(
        status="healthy",
        models_loaded=pipeline._is_initialized,
        index_size=pipeline.vector_store.size,
        identity_count=identity_mgr.count(),
        uptime_seconds=get_uptime(),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(
    pipeline: RecognitionPipeline = Depends(get_pipeline),
) -> ReadinessResponse:
    """Readiness probe — are all models loaded and warm?"""
    checks = {
        "pipeline_initialized": pipeline._is_initialized,
        "index_available": True,
    }
    return ReadinessResponse(
        ready=all(checks.values()),
        checks=checks,
    )


@router.get("/health/live")
async def liveness_check() -> dict:
    """Liveness probe — is the service responsive?"""
    return {"status": "alive"}
