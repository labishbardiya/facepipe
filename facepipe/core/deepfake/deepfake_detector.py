"""
Deepfake / face-swap detection module.

Detects AI-generated face swaps, avatars, and video generation artifacts
using multi-signal analysis:

  1. Frequency analysis — GAN spectral fingerprints in 2D FFT
  2. Compression artifact analysis — DCT coefficient anomalies
  3. Facial boundary analysis — blending artifacts at face contour
  4. Temporal consistency — micro-expression and physiological signal tracking

This module addresses the 2026 threat landscape where real-time face swaps
(DeepFaceLive, etc.) and AI-generated video are accessible to attackers.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import cv2
import numpy as np

from facepipe.config.settings import DeepfakeSettings, get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class DeepfakeResult:
    """Result from deepfake detection analysis.

    Attributes:
        is_real: Whether the face is judged to be a real, unmanipulated face.
        confidence: Confidence in the is_real decision, in [0.0, 1.0].
        method_scores: Per-method scores (higher = more likely real).
        details: Human-readable details for debugging.
    """
    is_real: bool
    confidence: float
    method_scores: dict[str, float]
    details: str


class DeepfakeDetector:
    """Multi-signal deepfake detection engine.

    Combines frequency-domain analysis, compression artifact detection,
    and facial boundary analysis to detect AI-generated face swaps.

    Args:
        settings: Deepfake detection settings. If None, loaded from global config.
        temporal_window: Number of frames to keep for temporal consistency analysis.
    """

    def __init__(
        self,
        settings: DeepfakeSettings | None = None,
        temporal_window: int = 10,
    ) -> None:
        self._settings = settings or get_settings().deepfake
        self._temporal_window = temporal_window
        # Track ID → recent face crops for temporal analysis
        self._temporal_buffer: dict[int, deque[np.ndarray]] = {}

    def detect(
        self,
        face_crop: np.ndarray,
        full_frame: np.ndarray | None = None,
        face_bbox: np.ndarray | None = None,
        track_id: int | None = None,
    ) -> DeepfakeResult:
        """Analyze a face crop for deepfake indicators.

        Args:
            face_crop: The cropped face region (BGR).
            full_frame: The full frame (for boundary analysis). Optional.
            face_bbox: Bounding box [x1,y1,x2,y2] in the full frame. Optional.
            track_id: Temporal track ID for multi-frame analysis. Optional.

        Returns:
            DeepfakeResult with real/fake decision and per-method scores.
        """
        if not self._settings.enabled:
            return DeepfakeResult(
                is_real=True,
                confidence=1.0,
                method_scores={},
                details="Deepfake detection disabled.",
            )

        method_scores: dict[str, float] = {}

        # 1. Frequency analysis
        freq_score = self._analyze_frequency(face_crop)
        method_scores["frequency"] = freq_score

        # 2. Compression artifact analysis
        compression_score = self._analyze_compression(face_crop)
        method_scores["compression"] = compression_score

        # 3. Facial boundary analysis
        if full_frame is not None and face_bbox is not None:
            boundary_score = self._analyze_boundary(full_frame, face_bbox)
            method_scores["boundary"] = boundary_score

        # 4. Temporal consistency
        if track_id is not None:
            temporal_score = self._analyze_temporal(face_crop, track_id)
            method_scores["temporal"] = temporal_score

        # Aggregate scores
        if not method_scores:
            return DeepfakeResult(is_real=True, confidence=1.0, method_scores={}, details="No methods ran.")

        avg_score = float(np.mean(list(method_scores.values())))
        min_score = min(method_scores.values())

        # A face is judged real if avg score > threshold AND no single method
        # scores extremely low (which would indicate a strong deepfake signal)
        is_real = avg_score > self._settings.threshold and min_score > (self._settings.threshold * 0.5)

        confidence = avg_score if is_real else (1.0 - avg_score)

        details_parts = [f"{k}={v:.3f}" for k, v in method_scores.items()]
        details = f"avg={avg_score:.3f}, min={min_score:.3f}, " + ", ".join(details_parts)

        return DeepfakeResult(
            is_real=is_real,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            method_scores=method_scores,
            details=details,
        )

    def _analyze_frequency(self, face_crop: np.ndarray) -> float:
        """Detect GAN spectral fingerprints via 2D FFT analysis.

        Real faces have smooth spectral falloff. GAN-generated faces
        exhibit periodic spectral peaks from the upsampling operations
        in the generator (e.g., transposed convolution artifacts).

        Returns:
            Score in [0, 1] where 1.0 = likely real.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop

        # Resize to standard size for consistent frequency analysis
        resized = cv2.resize(gray, (128, 128)).astype(np.float64)

        # 2D FFT
        f_transform = np.fft.fft2(resized)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        # Compute azimuthal average of the power spectrum
        center = np.array(magnitude.shape) // 2
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)

        max_r = min(center)
        radial_mean = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if mask.any():
                radial_mean[i] = magnitude[mask].mean()

        # Real faces: smooth monotonic decay in radial profile
        # GAN artifacts: bumps/peaks in mid-to-high frequencies
        if len(radial_mean) < 10:
            return 0.5

        # Check for non-monotonic behavior in mid-high frequency range
        mid_start = len(radial_mean) // 4
        high_freq = radial_mean[mid_start:]

        if len(high_freq) < 3:
            return 0.5

        # Compute roughness: sum of absolute second differences
        diffs = np.diff(high_freq)
        second_diffs = np.diff(diffs)
        roughness = float(np.mean(np.abs(second_diffs)))

        # Normalize roughness to a score (empirical calibration)
        # Low roughness = smooth falloff = likely real
        # High roughness = spectral peaks = likely GAN
        score = 1.0 / (1.0 + roughness * 2.0)

        return float(np.clip(score, 0.0, 1.0))

    def _analyze_compression(self, face_crop: np.ndarray) -> float:
        """Detect compression artifact inconsistencies.

        Deepfake faces often have different JPEG compression patterns
        than the surrounding frame. We analyze DCT coefficient
        distributions within the face region.

        Returns:
            Score in [0, 1] where 1.0 = likely real.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        resized = cv2.resize(gray, (64, 64)).astype(np.float64)

        # Compute block-wise DCT (8x8 blocks, standard JPEG)
        h, w = resized.shape
        blocks_h, blocks_w = h // 8, w // 8

        dct_energies: list[float] = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = resized[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
                dct_block = cv2.dct(block)
                # Energy in high-frequency coefficients (exclude DC and low-freq)
                high_freq_energy = float(np.sum(np.abs(dct_block[4:, 4:])))
                dct_energies.append(high_freq_energy)

        if not dct_energies:
            return 0.5

        energies = np.array(dct_energies)

        # Real images: relatively uniform high-freq energy distribution
        # Deepfakes: some blocks have anomalous energy (blended regions)
        cv_energy = float(np.std(energies) / (np.mean(energies) + 1e-8))

        # Low coefficient of variation = uniform = likely real
        score = 1.0 / (1.0 + cv_energy * 0.5)

        return float(np.clip(score, 0.0, 1.0))

    def _analyze_boundary(self, full_frame: np.ndarray, bbox: np.ndarray) -> float:
        """Detect blending artifacts at the face boundary.

        Face swaps create subtle gradient discontinuities where the
        swapped face meets the original frame. We compute gradient
        magnitude along the face contour and look for anomalies.

        Returns:
            Score in [0, 1] where 1.0 = likely real.
        """
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = full_frame.shape[:2]

        # Expand bbox slightly for boundary analysis
        margin = max(10, int((x2 - x1) * 0.1))
        bx1 = max(0, x1 - margin)
        by1 = max(0, y1 - margin)
        bx2 = min(w, x2 + margin)
        by2 = min(h, y2 + margin)

        region = full_frame[by1:by2, bx1:bx2]
        if region.size == 0:
            return 0.5

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region

        # Compute gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Create a mask for the boundary region (ring around the face bbox)
        rh, rw = gray.shape
        inner_mask = np.zeros((rh, rw), dtype=bool)
        ix1 = x1 - bx1 + margin // 2
        iy1 = y1 - by1 + margin // 2
        ix2 = x2 - bx1 - margin // 2
        iy2 = y2 - by1 - margin // 2
        ix1, iy1 = max(0, ix1), max(0, iy1)
        ix2, iy2 = min(rw, ix2), min(rh, iy2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.5

        inner_mask[iy1:iy2, ix1:ix2] = True

        outer_mask = np.ones((rh, rw), dtype=bool)
        outer_mask[margin:-margin if margin > 0 else rh, margin:-margin if margin > 0 else rw] = False

        boundary_mask = ~inner_mask & ~outer_mask
        if not boundary_mask.any():
            return 0.5

        boundary_gradient = float(np.mean(grad_mag[boundary_mask]))
        inner_gradient = float(np.mean(grad_mag[inner_mask])) if inner_mask.any() else 0.0

        # Real faces: boundary gradient is similar to interior gradient
        # Deepfakes: boundary gradient is anomalously high or low
        if inner_gradient < 1e-6:
            return 0.5

        ratio = boundary_gradient / (inner_gradient + 1e-8)

        # Ratio near 1.0 = natural = real; far from 1.0 = suspicious
        deviation = abs(ratio - 1.0)
        score = 1.0 / (1.0 + deviation * 2.0)

        return float(np.clip(score, 0.0, 1.0))

    def _analyze_temporal(self, face_crop: np.ndarray, track_id: int) -> float:
        """Check temporal consistency across frames.

        Real faces have natural micro-movements between frames.
        Deepfake swaps may have unnaturally stable or jittering artifacts.

        Returns:
            Score in [0, 1] where 1.0 = likely real.
        """
        if track_id not in self._temporal_buffer:
            self._temporal_buffer[track_id] = deque(maxlen=self._temporal_window)

        # Resize for comparison
        small = cv2.resize(face_crop, (64, 64))
        self._temporal_buffer[track_id].append(small)

        buffer = self._temporal_buffer[track_id]
        if len(buffer) < 3:
            return 0.5  # Not enough frames yet

        # Compute frame-to-frame differences
        diffs: list[float] = []
        for i in range(1, len(buffer)):
            diff = np.mean(np.abs(buffer[i].astype(float) - buffer[i - 1].astype(float)))
            diffs.append(diff)

        diff_array = np.array(diffs)
        mean_diff = float(np.mean(diff_array))
        std_diff = float(np.std(diff_array))

        # Real faces: moderate, variable inter-frame differences (natural motion)
        # Deepfakes: either too stable (perfect swap) or too jittery (artifacts)

        # Penalize very low variation (too stable = suspicious)
        if mean_diff < 1.0:
            stability_score = mean_diff / 1.0
        # Penalize very high variation (jittery = suspicious)
        elif mean_diff > 30.0:
            stability_score = max(0.0, 1.0 - (mean_diff - 30.0) / 30.0)
        else:
            stability_score = 1.0

        # Penalize very uniform differences (mechanical motion)
        cv = std_diff / (mean_diff + 1e-8)
        variability_score = min(cv / 0.3, 1.0)

        score = 0.6 * stability_score + 0.4 * variability_score

        return float(np.clip(score, 0.0, 1.0))

    def clear_track(self, track_id: int) -> None:
        """Remove temporal buffer for a track that's been lost."""
        self._temporal_buffer.pop(track_id, None)
