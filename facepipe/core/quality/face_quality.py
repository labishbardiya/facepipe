"""
Face Quality Assessment module.

Evaluates detected faces across multiple quality dimensions:
  - Blur detection (Laplacian variance)
  - Pose estimation (yaw/pitch from 5-point landmarks)
  - Illumination analysis (luminance mean/std)
  - Face size validation (relative to frame)
  - Composite quality scoring with configurable thresholds

Produces a QualityReport with per-check scores and pass/fail decisions
for both enrollment (strict) and recognition (relaxed) contexts.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import cv2
import numpy as np

from facepipe.config.settings import get_settings, QualitySettings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)

# Canonical 5-point landmark positions for a frontal face (normalized to 112x112)
# [left_eye, right_eye, nose, left_mouth, right_mouth]
_CANONICAL_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class QualityReport:
    """Results from face quality assessment.

    All individual scores are in [0.0, 1.0] where 1.0 is best quality.

    Attributes:
        blur_score: Sharpness score. Low = blurry.
        pose_score: Frontality score. Low = extreme angle.
        illumination_score: Lighting quality. Low = too dark or too bright.
        size_score: Face size adequacy. Low = too small.
        composite_score: Weighted combination of all scores.
        yaw_degrees: Estimated yaw in degrees (negative = left, positive = right).
        pitch_degrees: Estimated pitch in degrees (negative = up, positive = down).
        blur_variance: Raw Laplacian variance (for debugging).
        luminance_mean: Mean Y-channel value.
        passes_enrollment: Whether this face meets enrollment quality standards.
        passes_recognition: Whether this face meets recognition quality standards.
        rejection_reasons: List of specific reasons for rejection (empty if passes).
    """
    blur_score: float
    pose_score: float
    illumination_score: float
    size_score: float
    composite_score: float
    yaw_degrees: float
    pitch_degrees: float
    blur_variance: float
    luminance_mean: float
    passes_enrollment: bool
    passes_recognition: bool
    rejection_reasons: list[str]


class FaceQualityAssessor:
    """Assesses face quality across multiple dimensions.

    Args:
        settings: Quality settings. If None, loaded from global config.
    """

    def __init__(self, settings: Optional[QualitySettings] = None) -> None:
        self._settings = settings or get_settings().quality

    def assess(
        self,
        crop: np.ndarray,
        landmarks: np.ndarray,
        frame_area_ratio: float,
    ) -> QualityReport:
        """Run all quality checks on a detected face.

        Args:
            crop: The face crop (BGR, any size).
            landmarks: 5-point facial landmarks as shape (5, 2).
            frame_area_ratio: Face bounding box area / frame area.

        Returns:
            A QualityReport with per-check scores and pass/fail.
        """
        rejection_reasons: list[str] = []

        # 1. Blur detection
        blur_var, blur_score = self._check_blur(crop)
        if blur_score < 0.3:
            rejection_reasons.append(f"blur (variance={blur_var:.1f})")

        # 2. Pose estimation
        yaw, pitch, pose_score = self._check_pose(landmarks)
        if pose_score < 0.3:
            rejection_reasons.append(f"pose (yaw={yaw:.1f}°, pitch={pitch:.1f}°)")

        # 3. Illumination
        lum_mean, illum_score = self._check_illumination(crop)
        if illum_score < 0.3:
            rejection_reasons.append(f"illumination (mean={lum_mean:.1f})")

        # 4. Size
        size_score = self._check_size(frame_area_ratio)
        if size_score < 0.3:
            rejection_reasons.append(f"size (ratio={frame_area_ratio:.4f})")

        # 5. Composite score
        composite = (
            0.30 * blur_score
            + 0.25 * pose_score
            + 0.25 * illum_score
            + 0.20 * size_score
        )

        passes_enrollment = composite >= self._settings.enrollment_threshold and len(rejection_reasons) == 0
        passes_recognition = composite >= self._settings.recognition_threshold

        return QualityReport(
            blur_score=blur_score,
            pose_score=pose_score,
            illumination_score=illum_score,
            size_score=size_score,
            composite_score=composite,
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            blur_variance=blur_var,
            luminance_mean=lum_mean,
            passes_enrollment=passes_enrollment,
            passes_recognition=passes_recognition,
            rejection_reasons=rejection_reasons,
        )

    def _check_blur(self, crop: np.ndarray) -> tuple[float, float]:
        """Compute blur score using Laplacian variance.

        Higher variance = sharper image. We map it to [0, 1] using a
        sigmoid centered at the configured threshold.

        Returns:
            (raw_variance, normalized_score)
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Sigmoid mapping: score = 1 / (1 + exp(-k * (var - threshold)))
        # k controls steepness; we use k = 0.05 for a smooth transition
        threshold = self._settings.blur_threshold
        k = 0.05
        score = 1.0 / (1.0 + math.exp(-k * (variance - threshold)))

        return variance, float(np.clip(score, 0.0, 1.0))

    def _check_pose(self, landmarks: np.ndarray) -> tuple[float, float, float]:
        """Estimate head pose (yaw, pitch) from 5-point landmarks.

        Uses geometric relationships between landmark positions to estimate
        yaw and pitch without requiring a full 3D model.

        Returns:
            (yaw_degrees, pitch_degrees, pose_score)
        """
        if landmarks.shape != (5, 2):
            return 0.0, 0.0, 0.5

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]

        # Inter-ocular distance (reference scale)
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 1e-6:
            return 0.0, 0.0, 0.5

        # Yaw estimation: nose horizontal offset from eye midpoint
        eye_center = (left_eye + right_eye) / 2.0
        nose_offset_x = (nose[0] - eye_center[0]) / eye_dist

        # Map offset to approximate yaw (empirical: offset of 0.5 ≈ 30°)
        yaw = float(nose_offset_x * 60.0)

        # Pitch estimation: vertical ratio of nose relative to eyes and mouth
        mouth_center = (left_mouth + right_mouth) / 2.0
        total_height = mouth_center[1] - eye_center[1]
        if abs(total_height) < 1e-6:
            pitch = 0.0
        else:
            nose_ratio = (nose[1] - eye_center[1]) / total_height
            # Canonical ratio is ~0.5; deviation maps to pitch
            pitch = float((nose_ratio - 0.5) * 60.0)

        # Score: penalize deviation from frontal
        max_yaw = self._settings.pose_max_yaw
        max_pitch = self._settings.pose_max_pitch

        yaw_penalty = min(abs(yaw) / max_yaw, 1.0)
        pitch_penalty = min(abs(pitch) / max_pitch, 1.0)

        pose_score = 1.0 - max(yaw_penalty, pitch_penalty)
        pose_score = float(np.clip(pose_score, 0.0, 1.0))

        return yaw, pitch, pose_score

    def _check_illumination(self, crop: np.ndarray) -> tuple[float, float]:
        """Check face illumination quality via Y-channel statistics.

        Returns:
            (mean_luminance, illumination_score)
        """
        if crop.size == 0:
            return 0.0, 0.0

        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float64)
        mean_lum = float(np.mean(y_channel))
        std_lum = float(np.std(y_channel))

        min_lum = self._settings.illumination_min
        max_lum = self._settings.illumination_max

        # Score based on how well luminance falls in the acceptable range
        if mean_lum < min_lum:
            lum_score = mean_lum / min_lum
        elif mean_lum > max_lum:
            lum_score = max(0.0, 1.0 - (mean_lum - max_lum) / (255.0 - max_lum))
        else:
            lum_score = 1.0

        # Penalize very low contrast (flat lighting)
        contrast_penalty = min(std_lum / 30.0, 1.0)
        score = lum_score * (0.7 + 0.3 * contrast_penalty)

        return mean_lum, float(np.clip(score, 0.0, 1.0))

    def _check_size(self, frame_area_ratio: float) -> float:
        """Score face size relative to frame area.

        Returns:
            Size score in [0.0, 1.0].
        """
        min_ratio = self._settings.min_face_size

        if frame_area_ratio >= min_ratio * 3:
            return 1.0
        elif frame_area_ratio >= min_ratio:
            # Linear interpolation from threshold to 3x threshold
            return (frame_area_ratio - min_ratio) / (min_ratio * 2)
        else:
            return max(0.0, frame_area_ratio / min_ratio * 0.3)
