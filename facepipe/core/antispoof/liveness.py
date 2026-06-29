"""
Anti-spoofing / liveness detection module.

Detects physical presentation attacks:
  - Printed photo attacks (LBP texture analysis)
  - Screen replay attacks (moiré pattern via FFT)
  - Video replay attacks (temporal edge consistency)

Configurable security levels (LOW / MEDIUM / HIGH) control which checks
run and how strict the thresholds are.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import cv2
import numpy as np

from facepipe.config.settings import AntispoofSettings, SecurityLevel, get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class LivenessResult:
    """Result from liveness / anti-spoofing analysis.

    Attributes:
        is_live: Whether the face is judged to be a live person.
        confidence: Confidence in the is_live decision, in [0.0, 1.0].
        method_scores: Per-method scores (higher = more likely live).
        security_level: The security level used for this assessment.
    """
    is_live: bool
    confidence: float
    method_scores: dict[str, float]
    security_level: str


class LivenessDetector:
    """Multi-method anti-spoofing engine.

    Args:
        settings: Anti-spoofing settings. If None, loaded from global config.
        temporal_window: Frames to accumulate for temporal analysis.
    """

    def __init__(
        self,
        settings: AntispoofSettings | None = None,
        temporal_window: int = 8,
    ) -> None:
        self._settings = settings or get_settings().antispoof
        self._temporal_window = temporal_window
        # Track ID → recent face edge maps for temporal analysis
        self._edge_buffer: dict[int, deque[np.ndarray]] = {}

    def check(
        self,
        face_crop: np.ndarray,
        track_id: int | None = None,
    ) -> LivenessResult:
        """Run liveness checks on a face crop.

        Args:
            face_crop: The cropped face region (BGR).
            track_id: Track ID for temporal analysis (optional).

        Returns:
            LivenessResult with live/spoof decision and per-method scores.
        """
        method_scores: dict[str, float] = {}
        level = self._settings.security_level

        # Always run: LBP texture analysis (catches printed photos)
        lbp_score = self._check_lbp_texture(face_crop)
        method_scores["lbp_texture"] = lbp_score

        # MEDIUM and HIGH: add moiré pattern detection
        if level in (SecurityLevel.MEDIUM, SecurityLevel.HIGH):
            moire_score = self._check_moire_pattern(face_crop)
            method_scores["moire_pattern"] = moire_score

        # HIGH only: add temporal edge consistency
        if level == SecurityLevel.HIGH and track_id is not None:
            temporal_score = self._check_temporal_edges(face_crop, track_id)
            method_scores["temporal_edges"] = temporal_score

        # Aggregate
        if not method_scores:
            return LivenessResult(is_live=True, confidence=1.0, method_scores={}, security_level=level.value)

        avg_score = float(np.mean(list(method_scores.values())))

        # Stricter thresholds at higher security levels
        threshold_map = {
            SecurityLevel.LOW: 0.40,
            SecurityLevel.MEDIUM: 0.50,
            SecurityLevel.HIGH: 0.60,
        }
        threshold = threshold_map.get(level, 0.50)

        is_live = avg_score > threshold
        confidence = avg_score if is_live else (1.0 - avg_score)

        return LivenessResult(
            is_live=is_live,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            method_scores=method_scores,
            security_level=level.value,
        )

    def _check_lbp_texture(self, face_crop: np.ndarray) -> float:
        """Detect printed photo attacks using Local Binary Pattern (LBP) analysis.

        Live faces have rich, varied micro-textures (pores, fine wrinkles).
        Printed photos have smoother, more uniform textures.

        We compute the LBP histogram and measure its entropy —
        high entropy = rich texture = likely live.

        Returns:
            Score in [0, 1] where 1.0 = likely live.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        resized = cv2.resize(gray, (128, 128))

        # Compute LBP (simplified: 8-neighbor, radius 1)
        lbp = self._compute_lbp(resized)

        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-8)

        # Entropy of histogram
        nonzero = hist[hist > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))

        # Live faces typically have entropy > 5.0
        # Printed photos typically have entropy < 4.5
        # Map to score via sigmoid centered at the threshold
        score = 1.0 / (1.0 + np.exp(-2.0 * (entropy - self._settings.lbp_threshold * 8.0)))

        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _compute_lbp(gray: np.ndarray) -> np.ndarray:
        """Compute basic 8-neighbor LBP pattern for each pixel."""
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)

        # 8 neighbors around each pixel
        padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)

        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            lbp |= ((neighbor >= gray).astype(np.uint8) << bit)

        return lbp

    def _check_moire_pattern(self, face_crop: np.ndarray) -> float:
        """Detect screen replay attacks via moiré pattern detection.

        Screens produce moiré interference patterns visible in the
        high-frequency spectrum of captured images. We detect these
        by analyzing the FFT for characteristic high-frequency energy peaks.

        Returns:
            Score in [0, 1] where 1.0 = likely live (no moiré).
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        resized = cv2.resize(gray, (128, 128)).astype(np.float64)

        # 2D FFT
        f_transform = np.fft.fft2(resized)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Separate low-freq (center) and high-freq (edges) regions
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        radius = min(center_y, center_x) // 3

        # Create circular masks
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        low_freq_energy = float(np.sum(magnitude[dist <= radius]))
        high_freq_energy = float(np.sum(magnitude[dist > radius * 2]))

        if low_freq_energy < 1e-8:
            return 0.5

        # Moiré patterns: anomalously high ratio of high-freq to low-freq
        high_freq_ratio = high_freq_energy / (low_freq_energy + high_freq_energy)

        # Live faces: high_freq_ratio typically < 0.15
        # Screen replay: high_freq_ratio typically > 0.25
        score = 1.0 - min(high_freq_ratio / self._settings.fft_threshold, 1.0)

        return float(np.clip(score, 0.0, 1.0))

    def _check_temporal_edges(self, face_crop: np.ndarray, track_id: int) -> float:
        """Detect video replay via temporal edge consistency.

        Real faces produce naturally varying edge patterns between frames.
        Video replays may show fixed edges from the playback device bezel
        or unnaturally consistent edge patterns.

        Returns:
            Score in [0, 1] where 1.0 = likely live.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        edges = cv2.Canny(cv2.resize(gray, (64, 64)), 50, 150)

        if track_id not in self._edge_buffer:
            self._edge_buffer[track_id] = deque(maxlen=self._temporal_window)

        self._edge_buffer[track_id].append(edges)
        buffer = self._edge_buffer[track_id]

        if len(buffer) < 3:
            return 0.5  # Need more frames

        # Compute edge similarity between consecutive frames
        similarities: list[float] = []
        for i in range(1, len(buffer)):
            intersection = np.sum(buffer[i] & buffer[i - 1])
            union = np.sum(buffer[i] | buffer[i - 1])
            if union > 0:
                similarities.append(intersection / union)

        if not similarities:
            return 0.5

        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))

        # Real faces: moderate edge similarity with variation (0.2-0.7 typical)
        # Video replay: very high edge similarity with low variation (device bezel)
        if mean_sim > 0.85 and std_sim < 0.05:
            score = 0.2  # Suspiciously stable = replay
        elif mean_sim < 0.1:
            score = 0.3  # Suspiciously unstable
        else:
            score = 0.7 + 0.3 * min(std_sim / 0.15, 1.0)

        return float(np.clip(score, 0.0, 1.0))

    def clear_track(self, track_id: int) -> None:
        """Remove temporal buffer for a lost track."""
        self._edge_buffer.pop(track_id, None)
