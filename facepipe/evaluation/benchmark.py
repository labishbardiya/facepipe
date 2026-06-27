"""
Benchmark harness for face recognition evaluation.

Supports standard academic benchmarks:
  - LFW (Labeled Faces in the Wild)
  - CFP-FP (Celebrities in Frontal-Profile)
  - AgeDB-30 (Cross-age verification)
  - CPLFW (Cross-Pose LFW)

Computes TAR@FAR, AUC, EER, accuracy, and generates comparison reports.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import cv2
import numpy as np
from sklearn.metrics import auc, roc_curve

from facepipe.core.alignment.face_align import align_face
from facepipe.core.detection.scrfd_detector import SCRFDDetector
from facepipe.core.recognition.adaface_recognizer import AdaFaceRecognizer
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class FailedPair:
    """A pair that failed embedding extraction during benchmark.

    Attributes:
        path_a: Path to the first image.
        path_b: Path to the second image.
        is_same: Whether the pair is a positive (same person) pair.
        failure_reason: Why the pair failed.
    """
    path_a: str
    path_b: str
    is_same: int
    failure_reason: str  # e.g. "file_missing", "no_face_detected", "zero_embedding"


@dataclasses.dataclass
class BenchmarkResult:
    """Results from a benchmark evaluation.

    Attributes:
        dataset: Name of the benchmark dataset.
        model_version: Model version tested.
        accuracy: Best accuracy at optimal threshold.
        threshold: Optimal threshold.
        auc_score: Area Under ROC Curve.
        eer: Equal Error Rate.
        tar_at_far: TAR at various FAR thresholds.
        total_pairs: Total pairs evaluated.
        timestamp: When the benchmark was run.
        latency_per_pair_ms: Average latency per pair.
        failed_pairs: List of pairs that failed extraction.
        failure_summary: Count of failures by reason.
    """
    dataset: str
    model_version: str
    accuracy: float
    threshold: float
    auc_score: float
    eer: float
    tar_at_far: dict[str, float]  # {"1e-1": 0.99, "1e-2": 0.98, ...}
    total_pairs: int
    timestamp: float
    latency_per_pair_ms: float
    failed_pairs: list[FailedPair] = dataclasses.field(default_factory=list)
    failure_summary: dict[str, int] = dataclasses.field(default_factory=dict)


class BenchmarkHarness:
    """Face recognition benchmark harness.

    Evaluates the recognition pipeline against standard benchmark
    datasets to measure accuracy and compare model versions.

    Args:
        detector: SCRFD detector instance. If None, creates one.
        recognizer: AdaFace recognizer instance. If None, creates one.
    """

    def __init__(
        self,
        detector: SCRFDDetector | None = None,
        recognizer: AdaFaceRecognizer | None = None,
    ) -> None:
        self._detector = detector or SCRFDDetector()
        self._recognizer = recognizer or AdaFaceRecognizer()

    def evaluate_pairs(
        self,
        pairs: list[tuple[str, str, int]],
        dataset_name: str = "custom",
    ) -> BenchmarkResult:
        """Evaluate a list of image pairs.

        Args:
            pairs: List of (path_a, path_b, is_same) tuples.
                   is_same=1 if same person, 0 if different.
            dataset_name: Name for reporting.

        Returns:
            BenchmarkResult with all metrics.
        """
        scores: list[float] = []
        labels: list[int] = []
        latencies: list[float] = []
        failures: list[FailedPair] = []
        failure_counts: dict[str, int] = {}

        for i, (path_a, path_b, is_same) in enumerate(pairs):
            start = time.perf_counter()

            emb_a = self._extract_embedding(path_a)
            emb_b = self._extract_embedding(path_b)

            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            if emb_a is not None and emb_b is not None:
                sim = float(np.dot(emb_a.flatten(), emb_b.flatten()))
                scores.append(sim)
                labels.append(is_same)
            else:
                # Classify failure reason
                reason = self._classify_failure(path_a, path_b, emb_a, emb_b)
                failures.append(FailedPair(
                    path_a=path_a, path_b=path_b,
                    is_same=is_same, failure_reason=reason,
                ))
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

            if (i + 1) % 500 == 0:
                logger.info("benchmark_progress", pairs_done=i + 1, total=len(pairs))

        if not scores:
            return BenchmarkResult(
                dataset=dataset_name,
                model_version=self._recognizer.model_version,
                accuracy=0.0, threshold=0.0, auc_score=0.0, eer=0.0,
                tar_at_far={}, total_pairs=0, timestamp=time.time(),
                latency_per_pair_ms=0.0,
                failed_pairs=failures,
                failure_summary=failure_counts,
            )

        scores_arr = np.array(scores)
        labels_arr = np.array(labels)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
        auc_score = float(auc(fpr, tpr))

        # Find EER (where FPR = 1 - TPR)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

        # Find best accuracy
        best_acc = 0.0
        best_threshold = 0.0
        for t in thresholds:
            predictions = (scores_arr >= t).astype(int)
            acc = float(np.mean(predictions == labels_arr))
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(t)

        # TAR@FAR at standard thresholds
        far_thresholds = [1e-1, 1e-2, 1e-3, 1e-4]
        tar_at_far: dict[str, float] = {}
        for far_t in far_thresholds:
            idx = np.searchsorted(fpr, far_t)
            if idx < len(tpr):
                tar_at_far[f"{far_t:.0e}"] = float(tpr[idx])
            else:
                tar_at_far[f"{far_t:.0e}"] = float(tpr[-1])

        avg_latency = float(np.mean(latencies)) if latencies else 0.0

        result = BenchmarkResult(
            dataset=dataset_name,
            model_version=self._recognizer.model_version,
            accuracy=best_acc,
            threshold=best_threshold,
            auc_score=auc_score,
            eer=eer,
            tar_at_far=tar_at_far,
            total_pairs=len(scores),
            timestamp=time.time(),
            latency_per_pair_ms=avg_latency,
            failed_pairs=failures,
            failure_summary=failure_counts,
        )

        logger.info(
            "benchmark_complete",
            dataset=dataset_name,
            accuracy=f"{best_acc:.4f}",
            auc=f"{auc_score:.4f}",
            eer=f"{eer:.4f}",
            failed=len(failures),
        )

        return result

    def _extract_embedding(self, image_path: str) -> np.ndarray | None:
        """Extract embedding from an image file."""
        if not os.path.exists(image_path):
            return None

        img = cv2.imread(image_path)
        if img is None:
            return None

        faces = self._detector.detect(img)
        if not faces:
            return None

        face = faces[0]
        aligned = align_face(img, face.landmarks)
        result = self._recognizer.extract(aligned)

        if np.linalg.norm(result.embedding) > 0:
            return result.embedding
        return None

    @staticmethod
    def _classify_failure(path_a: str, path_b: str,
                          emb_a: np.ndarray | None,
                          emb_b: np.ndarray | None) -> str:
        """Classify why a pair failed embedding extraction."""
        reasons = []
        for path, emb in [(path_a, emb_a), (path_b, emb_b)]:
            if emb is not None:
                continue
            if not os.path.exists(path):
                reasons.append("file_missing")
            elif cv2.imread(path) is None:
                reasons.append("read_failure")
            else:
                # File exists and is readable — detection or extraction failed
                reasons.append("no_face_detected")
        return reasons[0] if reasons else "unknown"

    @staticmethod
    def parse_lfw_pairs(pairs_file: str, lfw_dir: str) -> list[tuple[str, str, int]]:
        """Parse LFW pairs.txt format.

        Format:
          Line 1: N (number of positive pairs)
          Positive: name\ttab\timg1\timg2
          Negative: name1\ttab\timg1\tname2\timg2
        """
        pairs: list[tuple[str, str, int]] = []

        with open(pairs_file) as f:
            lines = f.readlines()

        # Line 1 contains fold info, e.g. "10 300"
        int(lines[0].strip().split()[0])

        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                # Positive pair: same person
                name = parts[0]
                idx_a = int(parts[1])
                idx_b = int(parts[2])
                path_a = os.path.join(lfw_dir, name, f"{name}_{idx_a:04d}.jpg")
                path_b = os.path.join(lfw_dir, name, f"{name}_{idx_b:04d}.jpg")
                pairs.append((path_a, path_b, 1))
            elif len(parts) == 4:
                # Negative pair: different people
                name_a = parts[0]
                idx_a = int(parts[1])
                name_b = parts[2]
                idx_b = int(parts[3])
                path_a = os.path.join(lfw_dir, name_a, f"{name_a}_{idx_a:04d}.jpg")
                path_b = os.path.join(lfw_dir, name_b, f"{name_b}_{idx_b:04d}.jpg")
                pairs.append((path_a, path_b, 0))

        return pairs

    @staticmethod
    def format_report(results: list[BenchmarkResult]) -> str:
        """Format benchmark results as a human-readable report."""
        lines = [
            "=" * 70,
            "BENCHMARK REPORT",
            "=" * 70,
            "",
        ]

        for r in results:
            lines.extend([
                f"Dataset: {r.dataset}",
                f"Model:   {r.model_version}",
                f"Pairs:   {r.total_pairs} (of {r.total_pairs + len(r.failed_pairs)} total, {len(r.failed_pairs)} failed)",
                "",
                f"  Accuracy:   {r.accuracy:.4f} (threshold={r.threshold:.4f})",
                f"  AUC:        {r.auc_score:.4f}",
                f"  EER:        {r.eer:.4f}",
                f"  Latency:    {r.latency_per_pair_ms:.1f} ms/pair",
                "",
                "  TAR@FAR:",
            ])

            for far, tar in sorted(r.tar_at_far.items()):
                lines.append(f"    FAR={far}:  TAR={tar:.4f}")

            if r.failure_summary:
                lines.extend(["", "  Failure Breakdown:"])
                for reason, count in sorted(r.failure_summary.items(), key=lambda x: -x[1]):
                    lines.append(f"    {reason:<25s}: {count}")

            lines.extend(["", "-" * 70, ""])

        return "\n".join(lines)

    @staticmethod
    def save_results(results: list[BenchmarkResult], path: str) -> None:
        """Save results to JSON file."""
        data = []
        for r in results:
            data.append({
                "dataset": r.dataset,
                "model_version": r.model_version,
                "accuracy": r.accuracy,
                "threshold": r.threshold,
                "auc_score": r.auc_score,
                "eer": r.eer,
                "tar_at_far": r.tar_at_far,
                "total_pairs": r.total_pairs,
                "timestamp": r.timestamp,
                "latency_per_pair_ms": r.latency_per_pair_ms,
                "failed_count": len(r.failed_pairs),
                "failure_summary": r.failure_summary,
                "failed_pairs": [
                    {
                        "path_a": fp.path_a,
                        "path_b": fp.path_b,
                        "is_same": fp.is_same,
                        "failure_reason": fp.failure_reason,
                    }
                    for fp in r.failed_pairs
                ],
            })

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def parse_agedb_pairs(pairs_file: str, agedb_dir: str) -> list[tuple[str, str, int]]:
        """Parse AgeDB-30 pairs file.

        Same tab-delimited format as LFW:
          Line 1: header (folds count)
          Positive: name\\timg1\\timg2
          Negative: name1\\timg1\\tname2\\timg2
        """
        return BenchmarkHarness._parse_lfw_style_pairs(pairs_file, agedb_dir)

    @staticmethod
    def parse_cplfw_pairs(pairs_file: str, cplfw_dir: str) -> list[tuple[str, str, int]]:
        """Parse CPLFW (Cross-Pose LFW) pairs file."""
        return BenchmarkHarness._parse_lfw_style_pairs(pairs_file, cplfw_dir)

    @staticmethod
    def parse_cfp_fp_pairs(pairs_file: str, cfp_dir: str) -> list[tuple[str, str, int]]:
        """Parse CFP-FP (Frontal-Profile) pairs file."""
        return BenchmarkHarness._parse_lfw_style_pairs(pairs_file, cfp_dir)

    @staticmethod
    def _parse_lfw_style_pairs(pairs_file: str, data_dir: str) -> list[tuple[str, str, int]]:
        """Generic parser for LFW-style tab-delimited pair files.

        Handles the standard format used by LFW, AgeDB-30, CPLFW, CFP-FP.
        """
        pairs: list[tuple[str, str, int]] = []

        with open(pairs_file) as f:
            lines = f.readlines()

        # Skip header line(s) — may be "10 300" or just a count
        start_idx = 0
        for i, line in enumerate(lines):
            parts = line.strip().split()
            try:
                # If line is purely numeric (header), skip it
                [int(p) for p in parts]
                start_idx = i + 1
            except ValueError:
                break

        for line in lines[start_idx:]:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            if len(parts) == 3:
                # Positive pair: same person
                name = parts[0]
                idx_a = int(parts[1])
                idx_b = int(parts[2])
                path_a = os.path.join(data_dir, name, f"{name}_{idx_a:04d}.jpg")
                path_b = os.path.join(data_dir, name, f"{name}_{idx_b:04d}.jpg")
                pairs.append((path_a, path_b, 1))
            elif len(parts) == 4:
                # Negative pair: different people
                name_a = parts[0]
                idx_a = int(parts[1])
                name_b = parts[2]
                idx_b = int(parts[3])
                path_a = os.path.join(data_dir, name_a, f"{name_a}_{idx_a:04d}.jpg")
                path_b = os.path.join(data_dir, name_b, f"{name_b}_{idx_b:04d}.jpg")
                pairs.append((path_a, path_b, 0))

        return pairs

    def evaluate_suite(
        self,
        datasets: dict[str, tuple[str, str]],
    ) -> list[BenchmarkResult]:
        """Run benchmarks across multiple datasets.

        Args:
            datasets: Dict mapping dataset_name → (pairs_file, data_dir).

        Returns:
            List of BenchmarkResult, one per dataset.
        """
        results: list[BenchmarkResult] = []

        parser_map = {
            "LFW": self.parse_lfw_pairs,
            "AgeDB-30": self.parse_agedb_pairs,
            "CPLFW": self.parse_cplfw_pairs,
            "CFP-FP": self.parse_cfp_fp_pairs,
        }

        for name, (pairs_file, data_dir) in datasets.items():
            parser = parser_map.get(name, self._parse_lfw_style_pairs)
            logger.info("benchmark_suite_starting", dataset=name)

            pairs = parser(pairs_file, data_dir)
            if not pairs:
                logger.warning("benchmark_suite_no_pairs", dataset=name)
                continue

            result = self.evaluate_pairs(pairs, dataset_name=name)
            results.append(result)

        return results

    @staticmethod
    def compute_fnmr_per_quality(
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        quality_scores: np.ndarray | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute FNMR broken down by quality tier.

        Buckets positive pairs into quality tiers and computes FNMR
        separately per bucket. This diagnoses exactly which failure
        mode each improvement is fixing.

        Args:
            scores: Similarity scores for each pair.
            labels: Ground-truth labels (1=same, 0=different).
            threshold: Decision threshold.
            quality_scores: Per-pair quality scores (average of both images).

        Returns:
            Dict with per-tier FNMR and counts:
            {"high": {"fnmr": 0.01, "count": 1000}, ...}
        """
        # Define quality tiers
        tiers = {
            "high": (0.7, 1.0),
            "medium": (0.45, 0.7),
            "low": (0.0, 0.45),
        }

        # Filter to positive pairs only (FNMR is for genuine pairs)
        pos_mask = labels == 1
        pos_scores = scores[pos_mask]

        if quality_scores is not None:
            pos_qualities = quality_scores[pos_mask]
        else:
            # If no quality scores, put everything in "unknown" tier
            return {
                "all": {
                    "fnmr": float(np.mean(pos_scores < threshold)) if len(pos_scores) > 0 else 0.0,
                    "count": int(len(pos_scores)),
                }
            }

        result: dict[str, dict[str, float]] = {}
        for tier_name, (low, high) in tiers.items():
            tier_mask = (pos_qualities >= low) & (pos_qualities < high)
            tier_scores = pos_scores[tier_mask]

            if len(tier_scores) > 0:
                fnmr = float(np.mean(tier_scores < threshold))
                result[tier_name] = {
                    "fnmr": fnmr,
                    "count": int(len(tier_scores)),
                    "mean_score": float(np.mean(tier_scores)),
                }
            else:
                result[tier_name] = {
                    "fnmr": 0.0,
                    "count": 0,
                    "mean_score": 0.0,
                }

        return result

    @staticmethod
    def format_suite_comparison(results: list[BenchmarkResult]) -> str:
        """Format a cross-benchmark comparison table."""
        if not results:
            return "No results to compare."

        lines = [
            "=" * 90,
            "CROSS-BENCHMARK COMPARISON",
            "=" * 90,
            "",
            f"  {'Dataset':<15s} {'Accuracy':>10s} {'AUC':>10s} {'EER':>10s} "
            f"{'TAR@1e-3':>10s} {'Pairs':>8s} {'Failed':>8s}",
            f"  {'-'*71}",
        ]

        for r in results:
            tar_1e3 = r.tar_at_far.get("1e-03", 0.0)
            lines.append(
                f"  {r.dataset:<15s} {r.accuracy:>10.4f} {r.auc_score:>10.4f} "
                f"{r.eer:>10.4f} {tar_1e3:>10.4f} {r.total_pairs:>8d} "
                f"{len(r.failed_pairs):>8d}"
            )

        lines.extend(["", "=" * 90])
        return "\n".join(lines)

