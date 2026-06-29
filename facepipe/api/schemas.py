"""
Pydantic schemas for API request/response models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────
# Enrollment
# ──────────────────────────────────────────────────────────────

class EnrollRequest(BaseModel):
    """Enrollment request with base64-encoded images."""
    name: str = Field(..., description="Display name for the identity.", min_length=1, max_length=200)
    images: list[str] = Field(..., description="List of base64-encoded JPEG/PNG images.", min_length=1, max_length=20)

class QualityReportResponse(BaseModel):
    blur_score: float
    pose_score: float
    illumination_score: float
    size_score: float
    composite_score: float
    passes_enrollment: bool
    rejection_reasons: list[str]

class EnrollResponse(BaseModel):
    success: bool
    identity_id: str
    name: str
    embeddings_stored: int
    rejected_count: int
    quality_reports: list[QualityReportResponse]
    message: str


# ──────────────────────────────────────────────────────────────
# Recognition
# ──────────────────────────────────────────────────────────────

class RecognizeRequest(BaseModel):
    """Recognition request with a single image."""
    image: str = Field(..., description="Base64-encoded JPEG/PNG image.")
    mode: str = Field(default="photo", description="Mode of operation: 'photo' or 'video'.")

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
    identity: str | None = None
    confidence: float
    is_recognized: bool
    decision: str
    openset_decision: str
    component_scores: ComponentScores
    top_matches: list[MatchCandidate]
    quality_composite: float
    is_live: bool
    is_real: bool
    bbox: list[float]
    active_learning_action: str

class RecognizeResponse(BaseModel):
    faces: list[FaceResult]
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
    identities: list[IdentityResponse]
    total: int

class IdentityUpdateRequest(BaseModel):
    name: str | None = None


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
    checks: dict[str, bool]


# ──────────────────────────────────────────────────────────────
# Metrics / Events
# ──────────────────────────────────────────────────────────────

class EventResponse(BaseModel):
    event_id: str
    timestamp: float
    event_type: str
    identity_id: str | None
    payload: str

class EventQueryResponse(BaseModel):
    events: list[EventResponse]
    total: int


# ──────────────────────────────────────────────────────────────
# Error
# ──────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
