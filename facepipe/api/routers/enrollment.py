"""
Enrollment API router.

POST /api/v1/enroll — Quality-gated face enrollment.
"""

from __future__ import annotations

import base64
import hashlib

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from facepipe.api.dependencies import (
    get_encrypted_store,
    get_event_store,
    get_identity_manager,
    get_pipeline,
)
from facepipe.api.schemas import EnrollRequest, EnrollResponse, QualityReportResponse
from facepipe.core.pipeline import RecognitionPipeline
from facepipe.storage.encrypted_store import EncryptedEmbeddingStore
from facepipe.storage.event_store import EventStore, EventType
from facepipe.storage.identity_manager import IdentityManager

router = APIRouter()


def _decode_image(b64: str) -> np.ndarray:
    """Decode a base64-encoded image to a numpy array (BGR)."""
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")
    return img


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    request: EnrollRequest,
    pipeline: RecognitionPipeline = Depends(get_pipeline),
    identity_mgr: IdentityManager = Depends(get_identity_manager),
    event_store: EventStore = Depends(get_event_store),
    enc_store: EncryptedEmbeddingStore = Depends(get_encrypted_store),
) -> EnrollResponse:
    """Enroll a new identity with quality-gated embedding extraction."""
    # Decode images
    frames = []
    for img_b64 in request.images:
        try:
            frames.append(_decode_image(img_b64))
        except Exception:
            continue

    if not frames:
        raise HTTPException(status_code=400, detail="No valid images provided.")

    # Run enrollment pipeline
    result = pipeline.enroll(name=request.name, frames=frames)

    if result.success:
        # Create identity record
        identity_mgr.create(
            name=request.name,
            embedding_count=result.embeddings_stored,
            model_version=pipeline._recognizer.model_version,
            identity_id=result.identity_id,
        )

        # Log event
        image_hash = hashlib.sha256(frames[0].tobytes()).hexdigest()[:16]
        event_store.append(
            EventType.IDENTITY_ENROLLED,
            identity_id=result.identity_id,
            payload={
                "name": request.name,
                "embeddings": result.embeddings_stored,
                "rejected": result.rejected_count,
            },
            image_hash=image_hash,
        )

    # Build response
    quality_reports = [
        QualityReportResponse(
            blur_score=q.blur_score,
            pose_score=q.pose_score,
            illumination_score=q.illumination_score,
            size_score=q.size_score,
            composite_score=q.composite_score,
            passes_enrollment=q.passes_enrollment,
            rejection_reasons=q.rejection_reasons,
        )
        for q in result.quality_reports
    ]

    return EnrollResponse(
        success=result.success,
        identity_id=result.identity_id,
        name=result.name,
        embeddings_stored=result.embeddings_stored,
        rejected_count=result.rejected_count,
        quality_reports=quality_reports,
        message=result.message,
    )
