"""
Failure analysis for face recognition benchmark.

Investigates pairs that failed embedding extraction during benchmark
evaluation. Diagnoses failure reasons (no face detected, image too small,
file missing, read failure) and computes quality metrics on failed images
to identify systematic patterns.
"""

from __future__ import annotations

import dataclasses
import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from facepipe.core.detection.scrfd_detector import SCRFDDetector
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


class FailureReason(str, Enum):
    """Categorization of why an image failed embedding extraction."""
    FILE_MISSING = "file_missing"
    READ_FAILURE = "read_failure"
    NO_FACE_DETECTED = "no_face_detected"
    IMAGE_TOO_SMALL = "image_too_small"
    EXTRACTION_FAILED = "extraction_failed"


@dataclasses.dataclass
class ImageDiagnostic:
    """Quality diagnostics for a single image.

    Attributes:
        path: Path to the image file.
        exists: Whether the file exists on disk.
        readable: Whether OpenCV could decode the image.
        resolution: (height, width) if readable.
        blur_score: Laplacian variance — lower means blurrier.
        brightness: Mean pixel intensity (0-255).
        face_detected: Whether SCRFD found at least one face.
        face_score: Detection confidence of the top face, if any.
        face_area_ratio: Face area as fraction of frame area.
        failure_reason: Why this image failed, if applicable.
    """
    path: str
    exists: bool
    readable: bool
    resolution: Optional[Tuple[int, int]] = None
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    face_detected: bool = False
    face_score: Optional[float] = None
    face_area_ratio: Optional[float] = None
    failure_reason: Optional[FailureReason] = None


@dataclasses.dataclass
class FailedPairRecord:
    """Record of a pair that failed during benchmark evaluation.

    Attributes:
        path_a: Path to the first image.
        path_b: Path to the second image.
        is_same: Whether the pair is a positive (same person) pair.
        diag_a: Diagnostics for image A.
        diag_b: Diagnostics for image B.
    """
    path_a: str
    path_b: str
    is_same: int
    diag_a: ImageDiagnostic
    diag_b: ImageDiagnostic


@dataclasses.dataclass
class FailureAnalysisReport:
    """Complete failure analysis report.

    Attributes:
        total_pairs: Total pairs in the benchmark.
        failed_pairs: Number of pairs that failed.
        success_pairs: Number of pairs that succeeded.
        failure_rate: Fraction of pairs that failed.
        failure_breakdown: Count of failures by reason.
        positive_failures: Failed pairs that were positive (same person).
        negative_failures: Failed pairs that were negative (different people).
        failed_records: Detailed records of each failed pair.
        success_sample_stats: Quality stats from a random sample of successes.
        failure_stats: Quality stats from all failures.
    """
    total_pairs: int
    failed_pairs: int
    success_pairs: int
    failure_rate: float
    failure_breakdown: Dict[str, int]
    positive_failures: int
    negative_failures: int
    failed_records: List[FailedPairRecord]
    success_sample_stats: Dict[str, float]
    failure_stats: Dict[str, float]


# Minimum face crop dimension (pixels) below which we flag IMAGE_TOO_SMALL
_MIN_FACE_DIM = 20


class FailureAnalyzer:
    """Analyzes benchmark failures to identify patterns.

    Runs detection-only passes on failed images to diagnose why
    embedding extraction failed, and compares quality metrics
    against a sample of successful images.

    Args:
        detector: SCRFD detector instance. If None, creates one.
    """

    def __init__(self, detector: Optional[SCRFDDetector] = None) -> None:
        self._detector = detector or SCRFDDetector()

    def diagnose_image(self, image_path: str) -> ImageDiagnostic:
        """Run full diagnostics on a single image.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            ImageDiagnostic with all available quality metrics.
        """
        diag = ImageDiagnostic(path=image_path, exists=False, readable=False)

        # Check file existence
        if not os.path.exists(image_path):
            diag.failure_reason = FailureReason.FILE_MISSING
            return diag
        diag.exists = True

        # Try to read image
        img = cv2.imread(image_path)
        if img is None:
            diag.failure_reason = FailureReason.READ_FAILURE
            return diag
        diag.readable = True

        # Basic image quality metrics
        h, w = img.shape[:2]
        diag.resolution = (h, w)

        # Blur score: Laplacian variance (higher = sharper)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diag.blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Brightness: mean intensity
        diag.brightness = float(gray.mean())

        # Check if image is too small for reliable detection
        if h < _MIN_FACE_DIM or w < _MIN_FACE_DIM:
            diag.failure_reason = FailureReason.IMAGE_TOO_SMALL
            return diag

        # Run face detection
        faces = self._detector.detect(img)
        if not faces:
            diag.failure_reason = FailureReason.NO_FACE_DETECTED
            return diag

        diag.face_detected = True
        diag.face_score = faces[0].score
        diag.face_area_ratio = faces[0].frame_area_ratio

        return diag

    def analyze_pairs(
        self,
        pairs: List[Tuple[str, str, int]],
        success_sample_size: int = 200,
    ) -> FailureAnalysisReport:
        """Analyze all pairs and separate failures from successes.

        For each pair, runs diagnostics on both images. A pair is
        considered "failed" if either image fails detection. Also
        samples successful pairs to compute comparative quality stats.

        Args:
            pairs: List of (path_a, path_b, is_same) tuples.
            success_sample_size: How many successful pairs to sample
                                  for quality comparison.

        Returns:
            FailureAnalysisReport with full diagnostics.
        """
        failed_records: List[FailedPairRecord] = []
        success_diagnostics: List[ImageDiagnostic] = []
        failure_diagnostics: List[ImageDiagnostic] = []

        positive_failures = 0
        negative_failures = 0

        for i, (path_a, path_b, is_same) in enumerate(pairs):
            diag_a = self.diagnose_image(path_a)
            diag_b = self.diagnose_image(path_b)

            a_failed = diag_a.failure_reason is not None
            b_failed = diag_b.failure_reason is not None

            if a_failed or b_failed:
                failed_records.append(FailedPairRecord(
                    path_a=path_a,
                    path_b=path_b,
                    is_same=is_same,
                    diag_a=diag_a,
                    diag_b=diag_b,
                ))
                if a_failed:
                    failure_diagnostics.append(diag_a)
                if b_failed:
                    failure_diagnostics.append(diag_b)

                if is_same == 1:
                    positive_failures += 1
                else:
                    negative_failures += 1
            else:
                # Successful pair — collect for comparison sample
                if len(success_diagnostics) < success_sample_size * 2:
                    success_diagnostics.extend([diag_a, diag_b])

            if (i + 1) % 500 == 0:
                logger.info(
                    "failure_analysis_progress",
                    pairs_done=i + 1,
                    total=len(pairs),
                    failures_so_far=len(failed_records),
                )

        # Build failure breakdown
        failure_breakdown: Dict[str, int] = {}
        for diag in failure_diagnostics:
            reason = diag.failure_reason.value if diag.failure_reason else "unknown"
            failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1

        # Compute aggregate quality stats
        failure_stats = self._compute_quality_stats(failure_diagnostics)
        success_stats = self._compute_quality_stats(success_diagnostics)

        report = FailureAnalysisReport(
            total_pairs=len(pairs),
            failed_pairs=len(failed_records),
            success_pairs=len(pairs) - len(failed_records),
            failure_rate=len(failed_records) / len(pairs) if pairs else 0.0,
            failure_breakdown=failure_breakdown,
            positive_failures=positive_failures,
            negative_failures=negative_failures,
            failed_records=failed_records,
            success_sample_stats=success_stats,
            failure_stats=failure_stats,
        )

        logger.info(
            "failure_analysis_complete",
            total_pairs=len(pairs),
            failed=len(failed_records),
            breakdown=failure_breakdown,
        )

        return report

    @staticmethod
    def _compute_quality_stats(diagnostics: List[ImageDiagnostic]) -> Dict[str, float]:
        """Compute aggregate quality statistics from a list of diagnostics."""
        if not diagnostics:
            return {}

        blur_scores = [d.blur_score for d in diagnostics if d.blur_score is not None]
        brightness_vals = [d.brightness for d in diagnostics if d.brightness is not None]
        resolutions = [d.resolution for d in diagnostics if d.resolution is not None]
        face_scores = [d.face_score for d in diagnostics if d.face_score is not None]

        stats: Dict[str, float] = {
            "count": float(len(diagnostics)),
        }

        if blur_scores:
            stats["blur_mean"] = float(np.mean(blur_scores))
            stats["blur_median"] = float(np.median(blur_scores))
            stats["blur_min"] = float(np.min(blur_scores))

        if brightness_vals:
            stats["brightness_mean"] = float(np.mean(brightness_vals))
            stats["brightness_min"] = float(np.min(brightness_vals))
            stats["brightness_max"] = float(np.max(brightness_vals))

        if resolutions:
            areas = [h * w for h, w in resolutions]
            stats["resolution_mean_area"] = float(np.mean(areas))
            stats["resolution_min_area"] = float(np.min(areas))
            heights = [h for h, _ in resolutions]
            widths = [w for _, w in resolutions]
            stats["resolution_mean_h"] = float(np.mean(heights))
            stats["resolution_mean_w"] = float(np.mean(widths))

        if face_scores:
            stats["face_score_mean"] = float(np.mean(face_scores))
            stats["face_score_min"] = float(np.min(face_scores))

        return stats

    @staticmethod
    def format_report(report: FailureAnalysisReport) -> str:
        """Format a failure analysis report as human-readable text."""
        lines = [
            "=" * 70,
            "FAILURE ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Total Pairs:      {report.total_pairs}",
            f"Successful Pairs: {report.success_pairs}",
            f"Failed Pairs:     {report.failed_pairs} ({report.failure_rate:.2%})",
            f"  Positive (same person) failures: {report.positive_failures}",
            f"  Negative (diff person) failures: {report.negative_failures}",
            "",
            "Failure Breakdown:",
        ]

        for reason, count in sorted(report.failure_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"  {reason:<25s}: {count}")

        lines.extend(["", "-" * 70, ""])

        # Compare quality stats
        if report.failure_stats and report.success_sample_stats:
            lines.extend([
                "Quality Comparison (Failures vs. Success Sample):",
                "",
                f"  {'Metric':<30s} {'Failures':>12s} {'Successes':>12s}",
                f"  {'-'*54}",
            ])
            all_keys = sorted(set(report.failure_stats.keys()) | set(report.success_sample_stats.keys()))
            for key in all_keys:
                if key == "count":
                    continue
                f_val = report.failure_stats.get(key)
                s_val = report.success_sample_stats.get(key)
                f_str = f"{f_val:.2f}" if f_val is not None else "N/A"
                s_str = f"{s_val:.2f}" if s_val is not None else "N/A"
                lines.append(f"  {key:<30s} {f_str:>12s} {s_str:>12s}")

        lines.extend(["", "-" * 70, ""])

        # List individual failures (first 20)
        lines.append("Failed Pair Details (showing first 20):")
        lines.append("")
        for i, rec in enumerate(report.failed_records[:20]):
            pair_type = "SAME" if rec.is_same else "DIFF"
            lines.append(f"  [{i+1}] ({pair_type})")
            lines.append(f"    A: {Path(rec.path_a).name}")
            if rec.diag_a.failure_reason:
                lines.append(f"       Reason: {rec.diag_a.failure_reason.value}")
                if rec.diag_a.resolution:
                    lines.append(f"       Resolution: {rec.diag_a.resolution[1]}x{rec.diag_a.resolution[0]}")
                if rec.diag_a.blur_score is not None:
                    lines.append(f"       Blur: {rec.diag_a.blur_score:.1f}  Brightness: {rec.diag_a.brightness:.1f}")
            else:
                lines.append(f"       OK (score={rec.diag_a.face_score:.3f})" if rec.diag_a.face_score else "       OK")
            lines.append(f"    B: {Path(rec.path_b).name}")
            if rec.diag_b.failure_reason:
                lines.append(f"       Reason: {rec.diag_b.failure_reason.value}")
                if rec.diag_b.resolution:
                    lines.append(f"       Resolution: {rec.diag_b.resolution[1]}x{rec.diag_b.resolution[0]}")
                if rec.diag_b.blur_score is not None:
                    lines.append(f"       Blur: {rec.diag_b.blur_score:.1f}  Brightness: {rec.diag_b.brightness:.1f}")
            else:
                lines.append(f"       OK (score={rec.diag_b.face_score:.3f})" if rec.diag_b.face_score else "       OK")
            lines.append("")

        if len(report.failed_records) > 20:
            lines.append(f"  ... and {len(report.failed_records) - 20} more.")

        lines.extend(["", "=" * 70])
        return "\n".join(lines)

    @staticmethod
    def save_report(report: FailureAnalysisReport, path: str) -> None:
        """Save failure analysis report to JSON."""

        def _diag_to_dict(d: ImageDiagnostic) -> dict:
            return {
                "path": d.path,
                "exists": d.exists,
                "readable": d.readable,
                "resolution": list(d.resolution) if d.resolution else None,
                "blur_score": d.blur_score,
                "brightness": d.brightness,
                "face_detected": d.face_detected,
                "face_score": d.face_score,
                "face_area_ratio": d.face_area_ratio,
                "failure_reason": d.failure_reason.value if d.failure_reason else None,
            }

        data = {
            "total_pairs": report.total_pairs,
            "failed_pairs": report.failed_pairs,
            "success_pairs": report.success_pairs,
            "failure_rate": report.failure_rate,
            "failure_breakdown": report.failure_breakdown,
            "positive_failures": report.positive_failures,
            "negative_failures": report.negative_failures,
            "success_sample_stats": report.success_sample_stats,
            "failure_stats": report.failure_stats,
            "failed_records": [
                {
                    "path_a": r.path_a,
                    "path_b": r.path_b,
                    "is_same": r.is_same,
                    "diag_a": _diag_to_dict(r.diag_a),
                    "diag_b": _diag_to_dict(r.diag_b),
                }
                for r in report.failed_records
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
