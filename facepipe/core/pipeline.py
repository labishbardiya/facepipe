"""
End-to-end recognition pipeline orchestrator.

Wires all pipeline components together in the correct order:
  SCRFD Detection → Quality Assessment → Deepfake Detection →
  Anti-Spoofing → Alignment → [Face Restoration] → AdaFace Recognition →
  FAISS HNSW Search → Open-Set Recognition → Temporal Tracking →
  Decision Fusion → Active Learning

Provides two main methods:
  - process_frame(): Full recognition pipeline for a single video frame
  - enroll(): Quality-gated enrollment of a new identity
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional

import numpy as np

from facepipe.config.settings import get_settings, Settings
from facepipe.core.alignment.face_align import align_face
from facepipe.core.antispoof.liveness import LivenessDetector, LivenessResult
from facepipe.core.clustering.identity_cluster import (
    EmbeddingCluster,
    IdentityClusterEngine,
    IdentityClusters,
)
from facepipe.core.deepfake.deepfake_detector import DeepfakeDetector, DeepfakeResult
from facepipe.core.detection.scrfd_detector import DetectedFace, SCRFDDetector
from facepipe.core.fusion.decision_engine import DecisionFusionEngine, DecisionResult
from facepipe.core.learning.active_learning import ActiveLearningGate, LearningDecision
from facepipe.core.quality.face_quality import FaceQualityAssessor, QualityReport
from facepipe.core.quality.face_restoration import FaceRestorer
from facepipe.core.recognition.adaface_recognizer import AdaFaceRecognizer, EmbeddingResult
from facepipe.core.recognition.template_aggregator import TemplateAggregator
from facepipe.core.search.faiss_store import FAISSStore
from facepipe.core.search.openset import OpenSetRecognizer, OpenSetResult
from facepipe.core.tracking.byte_tracker import ByteTracker, TrackedFace
from facepipe.observability.logging import get_logger
from facepipe.observability.metrics import get_metrics

logger = get_logger(__name__)


@dataclasses.dataclass
class RecognitionResult:
    """Complete result for a single recognized face in a frame.

    Attributes:
        track_id: Persistent track ID across frames.
        bbox: Bounding box [x1, y1, x2, y2] in the original frame.
        identity: Recognized identity ID (None if unknown).
        decision: Full decision result from fusion engine.
        quality: Quality assessment report.
        liveness: Liveness check result.
        deepfake: Deepfake detection result.
        openset: Open-set recognition result.
        learning: Active learning recommendation.
        embedding: The extracted embedding (for caching/storage).
        latency_ms: Pipeline latency for this face in milliseconds.
    """
    track_id: int
    bbox: np.ndarray
    identity: Optional[str]
    decision: DecisionResult
    quality: QualityReport
    liveness: LivenessResult
    deepfake: DeepfakeResult
    openset: OpenSetResult
    learning: LearningDecision
    embedding: Optional[EmbeddingResult]
    latency_ms: float


@dataclasses.dataclass
class EnrollmentResult:
    """Result from an enrollment attempt.

    Attributes:
        success: Whether enrollment succeeded.
        identity_id: The identity ID (UUID) assigned.
        name: The display name.
        embeddings_stored: Number of embeddings successfully stored.
        quality_reports: Quality reports for each captured frame.
        rejected_count: Number of frames rejected by quality gate.
        message: Human-readable status message.
    """
    success: bool
    identity_id: str
    name: str
    embeddings_stored: int
    quality_reports: List[QualityReport]
    rejected_count: int
    message: str


class RecognitionPipeline:
    """End-to-end facial recognition pipeline.

    Orchestrates all pipeline components in sequence. Supports both
    single-frame processing (API/video) and enrollment workflows.

    Args:
        settings: Global settings. If None, loaded from global config.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._metrics = get_metrics()

        # Initialize components (lazy loading where possible)
        self._detector = SCRFDDetector()
        self._quality = FaceQualityAssessor()
        self._deepfake = DeepfakeDetector()
        self._liveness = LivenessDetector()
        self._recognizer = AdaFaceRecognizer()
        self._restorer = FaceRestorer()
        self._template_agg = TemplateAggregator()
        self._vector_store = FAISSStore(dim=self._settings.recognition.embedding_dim)
        self._openset = OpenSetRecognizer()
        self._cluster_engine = IdentityClusterEngine()
        self._tracker = ByteTracker()
        self._fusion = DecisionFusionEngine()
        self._active_learning = ActiveLearningGate()

        # In-memory identity clusters (loaded from facepipe.storage)
        self._identity_clusters: Dict[str, IdentityClusters] = {}

        self._is_initialized = False

    def initialize(self, index_path: Optional[str] = None) -> None:
        """Initialize the pipeline: load models, index, and identity data.

        Args:
            index_path: Path to saved FAISS index directory. If None,
                        uses default from settings.
        """
        if self._is_initialized:
            return

        logger.info("pipeline_initializing")

        # Load FAISS index if it exists
        if index_path is None:
            index_path = str(self._settings.data_dir / "index")

        try:
            self._vector_store.load(index_path)
            logger.info("index_loaded", size=self._vector_store.size)
        except Exception as e:
            logger.warning("index_load_failed", error=str(e))

        self._is_initialized = True
        logger.info("pipeline_initialized")

    def warmup(self) -> None:
        """Warmup all models with dummy data."""
        self.initialize()
        self._detector.warmup()
        self._recognizer.warmup()
        logger.info("pipeline_warmup_complete")

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "default",
    ) -> List[RecognitionResult]:
        """Process a single video frame through the full pipeline.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            camera_id: Camera identifier for multi-camera setups.

        Returns:
            List of RecognitionResult objects, one per detected face.
        """
        self.initialize()

        results: List[RecognitionResult] = []

        with self._metrics.pipeline_latency():
            # 1. Detection
            with self._metrics.detection_latency():
                detected_faces = self._detector.detect(frame)

            if not detected_faces:
                # Update tracker with no detections
                self._tracker.update([])
                return []

            # 2. Update tracker with detections
            detections = [(face.bbox.astype(np.float64), face.score) for face in detected_faces]
            active_tracks = self._tracker.update(detections)

            # Map tracks to detected faces by bbox IoU
            track_face_pairs = self._match_tracks_to_detections(active_tracks, detected_faces)

            # 3. Process each tracked face
            for track, face in track_face_pairs:
                face_start = time.perf_counter()

                # Check if this track needs recognition
                if not self._tracker.needs_recognition(track):
                    # Use cached result
                    cached = track.recognition_result
                    if cached is not None:
                        cached.track_id = track.track_id
                        cached.bbox = track.bbox
                        results.append(cached)
                        continue

                # 3a. Quality assessment
                with self._metrics.quality_latency():
                    quality = self._quality.assess(
                        face.crop, face.landmarks, face.frame_area_ratio
                    )

                if not quality.passes_recognition:
                    self._metrics.quality_rejections.labels(
                        reason=quality.rejection_reasons[0] if quality.rejection_reasons else "composite"
                    ).inc()
                    result = self._make_rejected_result(
                        track, quality, "quality_rejected"
                    )
                    results.append(result)
                    continue

                # 3b. Deepfake detection
                with self._metrics.deepfake_latency():
                    deepfake = self._deepfake.detect(
                        face.crop, frame, face.bbox, track.track_id
                    )

                if not deepfake.is_real:
                    self._metrics.deepfake_detections.labels(method="composite").inc()
                    self._metrics.recognition_total.labels(result="deepfake").inc()
                    result = self._make_rejected_result(
                        track, quality, "deepfake_rejected", deepfake=deepfake
                    )
                    results.append(result)
                    continue

                # 3c. Anti-spoofing
                with self._metrics.antispoof_latency():
                    liveness = self._liveness.check(face.crop, track.track_id)

                if not liveness.is_live:
                    self._metrics.spoof_detections.labels(method="composite").inc()
                    self._metrics.recognition_total.labels(result="spoof").inc()
                    result = self._make_rejected_result(
                        track, quality, "spoof_rejected", liveness=liveness, deepfake=deepfake
                    )
                    results.append(result)
                    continue

                # 3d. Alignment
                aligned = align_face(frame, face.landmarks)

                # 3e. Face restoration (quality-gated)
                restoration = self._restorer.restore(aligned, quality.composite_score)
                if restoration.was_restored:
                    # Re-align after restoration (geometry may have shifted)
                    restored_faces = self._detector.detect(restoration.restored)
                    if restored_faces:
                        aligned = align_face(restoration.restored, restored_faces[0].landmarks)
                    else:
                        aligned = restoration.restored

                # 3f. Recognition (embedding extraction)
                with self._metrics.recognition_latency():
                    emb_result = self._recognizer.extract(aligned)

                # 3g. Vector search
                with self._metrics.search_latency():
                    search_results = self._vector_store.search(
                        emb_result.embedding,
                        k=self._settings.openset.top_k,
                    )

                # 3h. Open-set recognition
                openset = self._openset.analyze(search_results)

                # 3i. Tracking consistency score
                tracking_consistency = min(track.frames_tracked / 30.0, 1.0)

                # 3j. Decision fusion
                decision = self._fusion.decide(
                    recognition_score=openset.best_score,
                    detection_score=face.score,
                    quality_score=quality.composite_score,
                    liveness_score=liveness.confidence,
                    tracking_consistency=tracking_consistency,
                    openset_margin=openset.margin,
                    deepfake_score=deepfake.confidence,
                    identity=openset.best_identity,
                    openset_decision=openset.decision,
                )

                # 3k. Active learning
                existing_centroids = None
                if decision.identity and decision.identity in self._identity_clusters:
                    existing_centroids = [
                        c.centroid for c in self._identity_clusters[decision.identity].clusters
                    ]

                learning = self._active_learning.evaluate(
                    identity_id=decision.identity,
                    confidence=decision.confidence,
                    embedding=emb_result.embedding,
                    existing_centroids=existing_centroids,
                    quality_score=quality.composite_score,
                )

                # Track metrics
                if decision.is_recognized:
                    self._metrics.recognition_total.labels(result="recognized").inc()
                elif openset.decision == "ambiguous":
                    self._metrics.recognition_total.labels(result="ambiguous").inc()
                else:
                    self._metrics.recognition_total.labels(result="unknown").inc()

                self._metrics.active_learning_actions.labels(action=learning.action).inc()

                latency_ms = (time.perf_counter() - face_start) * 1000

                result = RecognitionResult(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    identity=decision.identity,
                    decision=decision,
                    quality=quality,
                    liveness=liveness,
                    deepfake=deepfake,
                    openset=openset,
                    learning=learning,
                    embedding=emb_result,
                    latency_ms=latency_ms,
                )

                # Cache result in tracker
                self._tracker.set_recognition(track.track_id, result)
                results.append(result)

        self._metrics.active_tracks.set(self._tracker.active_count)
        return results

    def enroll(
        self,
        name: str,
        frames: List[np.ndarray],
        identity_id: Optional[str] = None,
    ) -> EnrollmentResult:
        """Enroll a new identity with quality-gated embedding extraction.

        Args:
            name: Display name for the identity.
            frames: List of BGR frames containing the face to enroll.
            identity_id: Optional UUID. If None, one is generated.

        Returns:
            EnrollmentResult with success status and details.
        """
        import ulid

        self.initialize()

        if identity_id is None:
            identity_id = str(ulid.new())

        quality_reports: List[QualityReport] = []
        accepted_embeddings: List[np.ndarray] = []
        accepted_qualities: List[float] = []
        accepted_norms: List[float] = []
        rejected_count = 0

        for frame in frames:
            # Detect
            faces = self._detector.detect(frame)
            if len(faces) != 1:
                rejected_count += 1
                continue

            face = faces[0]

            # Quality check (enrollment-grade)
            quality = self._quality.assess(face.crop, face.landmarks, face.frame_area_ratio)
            quality_reports.append(quality)

            if not quality.passes_enrollment:
                rejected_count += 1
                self._metrics.quality_rejections.labels(
                    reason=quality.rejection_reasons[0] if quality.rejection_reasons else "composite"
                ).inc()
                continue

            # Deepfake check
            deepfake = self._deepfake.detect(face.crop, frame, face.bbox)
            if not deepfake.is_real:
                rejected_count += 1
                continue

            # Liveness check
            liveness = self._liveness.check(face.crop)
            if not liveness.is_live:
                rejected_count += 1
                continue

            # Alignment + embedding
            aligned = align_face(frame, face.landmarks)
            emb_result = self._recognizer.extract(aligned)

            if np.linalg.norm(emb_result.embedding) > 0:
                accepted_embeddings.append(emb_result.embedding)
                accepted_qualities.append(quality.composite_score)
                accepted_norms.append(emb_result.raw_norm)

        if not accepted_embeddings:
            self._metrics.enrollment_total.labels(result="quality_rejected").inc()
            return EnrollmentResult(
                success=False,
                identity_id=identity_id,
                name=name,
                embeddings_stored=0,
                quality_reports=quality_reports,
                rejected_count=rejected_count,
                message=f"No frames passed quality checks ({rejected_count} rejected).",
            )

        # Template aggregation: build a quality-weighted template
        agg_result = self._template_agg.aggregate(
            embeddings=accepted_embeddings,
            quality_scores=accepted_qualities,
            raw_norms=accepted_norms,
        )
        logger.info(
            "enrollment_template_aggregated",
            identity=identity_id,
            inputs=agg_result.num_inputs,
            used=agg_result.num_used,
            rejected=agg_result.num_rejected,
            strategy=agg_result.strategy,
        )

        # Build clusters from individual embeddings (for appearance variants)
        clusters = self._cluster_engine.compute_clusters(
            accepted_embeddings, accepted_qualities,
        )
        self._identity_clusters[identity_id] = IdentityClusters(
            identity_id=identity_id,
            clusters=clusters,
            all_embeddings=accepted_embeddings,
            all_qualities=accepted_qualities,
        )

        # Add centroids to vector store
        centroid_ids = [identity_id] * len(clusters)
        centroid_embs = np.stack([c.centroid for c in clusters]).astype(np.float32)
        self._vector_store.add(centroid_ids, centroid_embs)

        # Save index
        index_path = str(self._settings.data_dir / "index")
        self._vector_store.save(index_path)

        self._metrics.enrollment_total.labels(result="success").inc()
        self._metrics.index_size.set(self._vector_store.size)
        self._metrics.identity_count.inc()

        return EnrollmentResult(
            success=True,
            identity_id=identity_id,
            name=name,
            embeddings_stored=len(accepted_embeddings),
            quality_reports=quality_reports,
            rejected_count=rejected_count,
            message=f"Enrolled '{name}' with {len(accepted_embeddings)} embeddings in {len(clusters)} clusters.",
        )

    def _match_tracks_to_detections(
        self,
        tracks: List[TrackedFace],
        detections: List[DetectedFace],
    ) -> List[tuple[TrackedFace, DetectedFace]]:
        """Match active tracks to detected faces by bbox proximity."""
        if not tracks or not detections:
            return []

        pairs: List[tuple[TrackedFace, DetectedFace]] = []
        used_dets = set()

        for track in tracks:
            best_det = None
            best_iou = 0.0

            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                iou = self._compute_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = (i, det)

            if best_det is not None and best_iou > 0.3:
                used_dets.add(best_det[0])
                pairs.append((track, best_det[1]))

        return pairs

    @staticmethod
    def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection

        return intersection / (union + 1e-8)

    def _make_rejected_result(
        self,
        track: TrackedFace,
        quality: QualityReport,
        reason: str,
        liveness: Optional[LivenessResult] = None,
        deepfake: Optional[DeepfakeResult] = None,
    ) -> RecognitionResult:
        """Create a RecognitionResult for a rejected face."""
        return RecognitionResult(
            track_id=track.track_id,
            bbox=track.bbox,
            identity=None,
            decision=DecisionResult(
                identity=None,
                confidence=0.0,
                is_recognized=False,
                component_scores={},
                decision_reason=reason,
                security_level="",
                active_learning_action="discard",
            ),
            quality=quality,
            liveness=liveness or LivenessResult(is_live=True, confidence=1.0, method_scores={}, security_level=""),
            deepfake=deepfake or DeepfakeResult(is_real=True, confidence=1.0, method_scores={}, details=""),
            openset=OpenSetResult(
                decision="unknown", top_matches=[], best_identity=None,
                best_score=0.0, margin=0.0, confidence=0.0,
                needs_verification=False, reason=reason,
            ),
            learning=LearningDecision(
                action="discard", identity_id=None, confidence=0.0,
                embedding=None, reason=reason, is_novel=False,
            ),
            embedding=None,
            latency_ms=0.0,
        )

    @property
    def vector_store(self) -> FAISSStore:
        """Access the vector store directly (for management operations)."""
        return self._vector_store

    @property
    def identity_clusters(self) -> Dict[str, IdentityClusters]:
        """Access identity clusters."""
        return self._identity_clusters
