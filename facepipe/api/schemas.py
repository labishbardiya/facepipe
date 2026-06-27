"""
Pydantic schemas for API request/response models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Enrollment
# ──────────────────────────────────────────────────────────────

class EnrollRequest(BaseModel):
    """Enrollment request with base64-encoded images."""
    name: str = Field(..., description="Display name for the identity.", min_length=1, max_length=200)
    images: List[str] = Field(..., description="List of base64-encoded JPEG/PNG images.", min_length=1, max_length=20)

class QualityReportResponse(BaseModel):
    blur_score: float
    pose_score: float
    illumination_score: float
    size_score: float
    composite_score: float
    passes_enrollment: bool
    rejection_reasons: List[str]

class EnrollResponse(BaseModel):
    success: bool
    identity_id: str
    name: str
    embeddings_stored: int
    rejected_count: int
    quality_reports: List[QualityReportResponse]
    message: str


# ──────────────────────────────────────────────────────────────
# Recognition
# ──────────────────────────────────────────────────────────────

class RecognizeRequest(BaseModel):
    """Recognition request with a single image."""
    image: str = Field(..., description="Base64-encoded JPEG/PNG image.")

class ComponentScores(BaseModel):
    recognition: float = 0.0
    detection: float = 0.0
    quality: float = 0.0
    liveness: float = 0.0
    tracking: float = 0.0
    openset_margin: float = 0.0
    deepfake: float = 0.0

class MatchCandidate(BaseModel):
    identity_id: str
    score: float
    rank: int

class FaceResult(BaseModel):
    identity: Optional[str] = None
    confidence: float
    is_recognized: bool
    decision: str
    openset_decision: str
    component_scores: ComponentScores
    top_matches: List[MatchCandidate]
    quality_composite: float
    is_live: bool
    is_real: bool
    bbox: List[float]
    active_learning_action: str

class RecognizeResponse(BaseModel):
    faces: List[FaceResult]
    face_count: int
    latency_ms: float


# ──────────────────────────────────────────────────────────────
# Identity Management
# ──────────────────────────────────────────────────────────────

class IdentityResponse(BaseModel):
    identity_id: str
    name: str
    created_at: float
    last_seen: float
    embedding_count: int
    cluster_count: int
    model_version: str
    is_active: bool

class IdentityListResponse(BaseModel):
    identities: List[IdentityResponse]
    total: int

class IdentityUpdateRequest(BaseModel):
    name: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., description="healthy / degraded / unhealthy")
    models_loaded: bool
    index_size: int
    identity_count: int
    uptime_seconds: float
    version: str = "2.0.0"

class ReadinessResponse(BaseModel):
    ready: bool
    checks: Dict[str, bool]


# ──────────────────────────────────────────────────────────────
# Metrics / Events
# ──────────────────────────────────────────────────────────────

class EventResponse(BaseModel):
    event_id: str
    timestamp: float
    event_type: str
    identity_id: Optional[str]
    payload: str

class EventQueryResponse(BaseModel):
    events: List[EventResponse]
    total: int


# ──────────────────────────────────────────────────────────────
# Error
# ──────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
