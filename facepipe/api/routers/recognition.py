"""
Recognition API router.

POST /api/v1/recognize — Full-pipeline face recognition.
"""

from __future__ import annotations

import base64
import hashlib
import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from facepipe.api.dependencies import get_event_store, get_pipeline
from facepipe.api.schemas import (
    ComponentScores,
    FaceResult,
    MatchCandidate,
    RecognizeRequest,
    RecognizeResponse,
)
from facepipe.core.pipeline import RecognitionPipeline
from facepipe.storage.event_store import EventStore, EventType

router = APIRouter()


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    request: RecognizeRequest,
    pipeline: RecognitionPipeline = Depends(get_pipeline),
    event_store: EventStore = Depends(get_event_store),
) -> RecognizeResponse:
    """Recognize faces in an image through the full pipeline."""
    # Decode image
    try:
        data = base64.b64decode(request.image)
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Decode failed")
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    start = time.perf_counter()

    # Run pipeline
    results = pipeline.process_frame(
        frame=frame,
        camera_id="api_single_image",
        mode=request.mode,
    )

    total_latency = (time.perf_counter() - start) * 1000

    # Build response
    face_results = []

    # Get identity manager to look up names
    from facepipe.api.dependencies import get_identity_manager
    identity_mgr = get_identity_manager()

    for r in results:
        identity_name = r.identity
        if r.identity:
            identity_record = identity_mgr.get(r.identity)
            if identity_record:
                identity_name = identity_record.name

        cs = r.decision.component_scores
        face_results.append(FaceResult(
            identity=identity_name,
            confidence=r.decision.confidence,
            is_recognized=r.decision.is_recognized,
            decision=r.decision.decision_reason,
            openset_decision=r.openset.decision,
            component_scores=ComponentScores(
                recognition=cs.get("recognition", 0.0),
                detection=cs.get("detection", 0.0),
                quality=cs.get("quality", 0.0),
                liveness=cs.get("liveness", 0.0),
                tracking=cs.get("tracking", 0.0),
                openset_margin=cs.get("openset_margin", 0.0),
                deepfake=cs.get("deepfake", 0.0),
            ),
            top_matches=[
                MatchCandidate(
                    identity_id=m.identity_id,
                    score=m.score,
                    rank=m.rank,
                )
                for m in r.openset.top_matches[:5]
            ],
            quality_composite=r.quality.composite_score,
            is_live=r.liveness.is_live,
            is_real=r.deepfake.is_real,
            bbox=r.bbox.tolist(),
            active_learning_action=r.learning.action,
        ))

        # Log recognition event
        image_hash = hashlib.sha256(frame.tobytes()).hexdigest()[:16]
        event_store.append(
            EventType.RECOGNITION_ATTEMPT,
            identity_id=r.identity,
            payload={
                "confidence": r.decision.confidence,
                "openset": r.openset.decision,
                "is_recognized": r.decision.is_recognized,
                "active_learning": r.learning.action,
            },
            image_hash=image_hash,
        )

    return RecognizeResponse(
        faces=face_results,
        face_count=len(face_results),
        latency_ms=total_latency,
    )
