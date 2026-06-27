"""
Centralized configuration for the facial recognition platform.

All settings are loaded from environment variables (prefixed FR_), .env files,
or fall back to defaults defined here. This is the single source of truth for
every tunable parameter in the system.
"""

from __future__ import annotations

import enum
import functools
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecurityLevel(enum.StrEnum):
    """Security level for anti-spoofing and fusion decision boundaries."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FusionSecurityLevel(enum.StrEnum):
    """Decision fusion security level controlling acceptance thresholds."""
    STANDARD = "STANDARD"
    ELEVATED = "ELEVATED"
    MAXIMUM = "MAXIMUM"


class RecognitionModel(enum.StrEnum):
    """Which recognition backbone to use."""
    ADAFACE = "adaface"
    ARCFACE = "arcface"


class KeyProviderType(enum.StrEnum):
    """How encryption keys are sourced."""
    ENV = "env"
    FILE = "file"


# ──────────────────────────────────────────────────────────────
# Sub-settings groups
# ──────────────────────────────────────────────────────────────


class DetectionSettings(BaseSettings):
    """SCRFD face detection parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_DET_")

    size: int = Field(default=640, description="Input size for detection model (square).")
    threshold: float = Field(default=0.5, description="Minimum detection confidence.")


class QualitySettings(BaseSettings):
    """Face quality assessment thresholds."""
    model_config = SettingsConfigDict(env_prefix="FR_QUALITY_")

    blur_threshold: float = Field(default=100.0, description="Laplacian variance below this = blurry.")
    pose_max_yaw: float = Field(default=30.0, description="Max absolute yaw in degrees.")
    pose_max_pitch: float = Field(default=25.0, description="Max absolute pitch in degrees.")
    illumination_min: float = Field(default=40.0, description="Min mean luminance (Y channel).")
    illumination_max: float = Field(default=220.0, description="Max mean luminance (Y channel).")
    min_face_size: float = Field(default=0.015, description="Min face area as fraction of frame area.")
    enrollment_threshold: float = Field(default=0.65, description="Min composite quality for enrollment.")
    recognition_threshold: float = Field(default=0.45, description="Min composite quality for recognition.")


class DeepfakeSettings(BaseSettings):
    """Deepfake detection parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_DEEPFAKE_")

    enabled: bool = Field(default=True, description="Whether to run deepfake detection.")
    threshold: float = Field(default=0.5, description="Confidence threshold for real vs fake.")


class AntispoofSettings(BaseSettings):
    """Anti-spoofing / liveness detection parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_ANTISPOOF_")

    security_level: SecurityLevel = Field(default=SecurityLevel.MEDIUM)
    lbp_threshold: float = Field(default=0.6, description="LBP texture analysis threshold.")
    fft_threshold: float = Field(default=0.5, description="Moiré pattern FFT threshold.")


class RecognitionSettings(BaseSettings):
    """Face recognition model parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_RECOGNITION_")

    model: RecognitionModel = Field(default=RecognitionModel.ADAFACE)
    embedding_dim: int = Field(default=512, description="Embedding dimensionality.")
    model_path: str = Field(default="", description="Path to ONNX model (empty = auto-download).")
    tta_enabled: bool = Field(default=True, description="Enable test-time augmentation (flip averaging).")
    tta_extended: bool = Field(default=False, description="Extended TTA: flip + brightness/contrast (4x cost).")


class SearchSettings(BaseSettings):
    """FAISS HNSW search index parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_HNSW_")

    m: int = Field(default=32, description="HNSW graph connectivity.")
    ef_construction: int = Field(default=200, description="Construction-time search depth.")
    ef_search: int = Field(default=64, description="Query-time search depth.")


class OpenSetSettings(BaseSettings):
    """Open-set recognition parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_OPENSET_")

    recognition_threshold: float = Field(default=0.4, description="Min similarity for recognition.")
    margin_threshold: float = Field(default=0.05, description="Min margin between top-1 and top-2.")
    top_k: int = Field(default=5, description="Number of candidates to retrieve.")


class ClusterSettings(BaseSettings):
    """Identity clustering parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_CLUSTER_")

    min_clusters: int = Field(default=2, description="Min clusters per identity.")
    max_clusters: int = Field(default=8, description="Max clusters per identity.")
    merge_threshold: float = Field(default=0.85, description="Cosine sim above which clusters merge.")
    new_cluster_threshold: float = Field(default=0.65, description="Cosine sim below which new cluster created.")


class TrackingSettings(BaseSettings):
    """ByteTrack temporal tracking parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_TRACK_")

    buffer: int = Field(default=30, description="Frames to keep lost tracks.")
    match_threshold: float = Field(default=0.7, description="IoU threshold for track matching.")
    re_recognize_interval: int = Field(default=30, description="Frames between re-recognition attempts.")
    high_det_threshold: float = Field(default=0.6, description="High-confidence detection threshold.")
    low_det_threshold: float = Field(default=0.1, description="Low-confidence detection threshold.")


class ActiveLearningSettings(BaseSettings):
    """Active learning gate parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_AL_")

    auto_add_threshold: float = Field(default=0.90, description="Above this = auto-add embedding.")
    verify_threshold: float = Field(default=0.55, description="Above this = request human verification.")
    max_auto_adds_per_hour: int = Field(default=10, description="Rate limit per identity.")
    novelty_threshold: float = Field(default=0.3, description="Min distance from existing to be 'novel'.")


class FusionSettings(BaseSettings):
    """Decision fusion engine parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_FUSION_")

    security_level: FusionSecurityLevel = Field(default=FusionSecurityLevel.STANDARD)
    weight_recognition: float = Field(default=0.35)
    weight_detection: float = Field(default=0.05)
    weight_quality: float = Field(default=0.10)
    weight_liveness: float = Field(default=0.15)
    weight_tracking: float = Field(default=0.10)
    weight_openset_margin: float = Field(default=0.15)
    weight_deepfake: float = Field(default=0.10)

    threshold_standard: float = Field(default=0.50)
    threshold_elevated: float = Field(default=0.65)
    threshold_maximum: float = Field(default=0.80)


class TemplateSettings(BaseSettings):
    """Template aggregation parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_TEMPLATE_")

    strategy: str = Field(default="quality_weighted", description="Aggregation strategy: quality_weighted | norm_weighted | top_k")
    min_frames: int = Field(default=3, description="Minimum frames before aggregating.")
    top_k: int = Field(default=5, description="Top-K highest-quality embeddings for top_k strategy.")
    outlier_sigma: float = Field(default=1.5, description="Outlier rejection: reject if sim < mean - sigma * std.")


class NormalizationSettings(BaseSettings):
    """Score normalization parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_NORM_")

    method: str = Field(default="z_norm", description="Normalization method: z_norm | t_norm | zt_norm | none")
    cohort_size: int = Field(default=200, description="Number of cohort embeddings for normalization.")


class EnsembleSettings(BaseSettings):
    """Multi-model ensemble parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_ENSEMBLE_")

    enabled: bool = Field(default=False, description="Enable multi-model ensemble.")
    models: list[str] = Field(default=["adaface", "arcface"], description="Models to ensemble.")
    fusion_strategy: str = Field(default="score_level", description="Fusion: concat_pca | score_level | quality_gated")


class StorageSettings(BaseSettings):
    """Storage and security parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_")

    encryption_key: str = Field(default="", description="Base64-encoded 32-byte key.")
    key_provider: KeyProviderType = Field(default=KeyProviderType.ENV)
    key_file_path: str = Field(default="", description="Path to key file (if provider=file).")
    image_retention_hours: int = Field(default=0, description="0 = no image storage.")
    event_retention_days: int = Field(default=90)


class APISettings(BaseSettings):
    """FastAPI server parameters."""
    model_config = SettingsConfigDict(env_prefix="FR_API_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    cors_origins: list[str] = Field(default=["*"])
    rate_limit: int = Field(default=100, description="Max requests per minute per client.")


# ──────────────────────────────────────────────────────────────
# Root settings — aggregates all sub-settings
# ──────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Root configuration object aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_prefix="FR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)

    # Paths
    data_dir: Path = Field(default=Path("./data"))
    models_dir: Path = Field(default=Path("./models"))
    legacy_db_dir: Path = Field(default=Path("./db"))

    # Sub-settings (instantiated eagerly)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    deepfake: DeepfakeSettings = Field(default_factory=DeepfakeSettings)
    antispoof: AntispoofSettings = Field(default_factory=AntispoofSettings)
    recognition: RecognitionSettings = Field(default_factory=RecognitionSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    openset: OpenSetSettings = Field(default_factory=OpenSetSettings)
    clustering: ClusterSettings = Field(default_factory=ClusterSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    active_learning: ActiveLearningSettings = Field(default_factory=ActiveLearningSettings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)
    template: TemplateSettings = Field(default_factory=TemplateSettings)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    ensemble: EnsembleSettings = Field(default_factory=EnsembleSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    api: APISettings = Field(default_factory=APISettings)

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "embeddings").mkdir(exist_ok=True)
        (self.data_dir / "events").mkdir(exist_ok=True)
        (self.data_dir / "index").mkdir(exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached)."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
