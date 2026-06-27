"""
SCRFD face detector wrapper.

Wraps InsightFace's SCRFD into a clean, typed interface with configurable
detection size, confidence threshold, and GPU/CPU/CoreML runtime selection.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from insightface.app import FaceAnalysis

from facepipe.config.settings import DetectionSettings, get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class DetectedFace:
    """A single face detected in a frame.

    Attributes:
        bbox: Bounding box as [x1, y1, x2, y2] in pixel coordinates.
        landmarks: 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth)
                   as shape (5, 2) array.
        score: Detection confidence in [0.0, 1.0].
        crop: The cropped face region from the original frame (BGR).
        frame_area_ratio: Face area as a fraction of total frame area.
    """
    bbox: np.ndarray
    landmarks: np.ndarray
    score: float
    crop: np.ndarray
    frame_area_ratio: float


class SCRFDDetector:
    """SCRFD face detector using InsightFace's buffalo_l model pack.

    The detector lazily initializes the model on first use and caches it
    for subsequent calls.

    Args:
        settings: Detection settings. If None, loaded from global config.
        ctx_id: ONNX Runtime device ID. -1 = CPU, 0+ = GPU index.
                Auto-detected from available providers if not specified.
    """

    def __init__(
        self,
        settings: DetectionSettings | None = None,
        ctx_id: int | None = None,
    ) -> None:
        self._settings = settings or get_settings().detection
        self._ctx_id = ctx_id if ctx_id is not None else self._auto_detect_device()
        self._app: FaceAnalysis | None = None

    @staticmethod
    def _auto_detect_device() -> int:
        """Auto-detect the best available device.

        Returns 0 if CUDA is available, -1 for CPU.
        CoreML is handled automatically by ONNX Runtime on macOS.
        """
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                return 0
            # CoreML and CPU are both handled with ctx_id=-1;
            # ONNX Runtime will prefer CoreML on Apple Silicon automatically
            return -1
        except ImportError:
            return -1

    def _ensure_loaded(self) -> FaceAnalysis:
        """Lazily initialize the InsightFace FaceAnalysis pipeline."""
        if self._app is None:
            det_size = (self._settings.size, self._settings.size)
            logger.info(
                "loading_scrfd",
                det_size=det_size,
                ctx_id=self._ctx_id,
            )
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                providers = [p for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"] if p in available] or ["CPUExecutionProvider"]
            except ImportError:
                providers = ["CPUExecutionProvider"]

            self._app = FaceAnalysis(name="buffalo_l", providers=providers)
            self._app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=det_size)
            logger.info("scrfd_loaded", providers=providers)
        return self._app

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect all faces in a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of DetectedFace objects, sorted by detection confidence (descending).
        """
        app = self._ensure_loaded()
        raw_faces = app.get(frame)

        frame_area = frame.shape[0] * frame.shape[1]
        results: list[DetectedFace] = []

        for face in raw_faces:
            if face.det_score < self._settings.threshold:
                continue

            bbox = face.bbox.astype(np.int32)
            x1, y1, x2, y2 = np.clip(bbox, 0, None)
            x2 = min(x2, frame.shape[1])
            y2 = min(y2, frame.shape[0])

            # Extract face crop
            crop = frame[y1:y2, x1:x2].copy() if (y2 > y1 and x2 > x1) else np.zeros((1, 1, 3), dtype=np.uint8)

            face_area = (x2 - x1) * (y2 - y1)
            area_ratio = face_area / frame_area if frame_area > 0 else 0.0

            landmarks = face.kps if face.kps is not None else np.zeros((5, 2), dtype=np.float32)

            results.append(DetectedFace(
                bbox=bbox,
                landmarks=landmarks,
                score=float(face.det_score),
                crop=crop,
                frame_area_ratio=area_ratio,
            ))

        # Sort by confidence descending
        results.sort(key=lambda f: f.score, reverse=True)
        return results

    def warmup(self) -> None:
        """Pre-load the model by running a dummy detection."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        logger.info("scrfd_warmup_complete")
