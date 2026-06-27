"""
Prometheus metrics for the facial recognition platform.

Provides histograms, counters, and gauges for every pipeline stage,
enabling real-time monitoring and alerting.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Counter, Gauge, Histogram, Info


class MetricsCollector:
    """Centralized Prometheus metrics for the facial recognition platform.

    Usage:
        metrics = MetricsCollector()

        with metrics.detection_latency():
            faces = detector.detect(frame)

        metrics.recognition_total.labels(result="recognized").inc()
    """

    def __init__(self) -> None:
        # ── Latency histograms ──
        self.detection_latency_hist = Histogram(
            "fr_detection_latency_seconds",
            "Face detection latency in seconds.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self.quality_latency_hist = Histogram(
            "fr_quality_latency_seconds",
            "Face quality assessment latency in seconds.",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
        )
        self.deepfake_latency_hist = Histogram(
            "fr_deepfake_latency_seconds",
            "Deepfake detection latency in seconds.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
        )
        self.antispoof_latency_hist = Histogram(
            "fr_antispoof_latency_seconds",
            "Anti-spoofing check latency in seconds.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
        )
        self.recognition_latency_hist = Histogram(
            "fr_recognition_latency_seconds",
            "Face recognition (embedding extraction) latency in seconds.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
        )
        self.search_latency_hist = Histogram(
            "fr_search_latency_seconds",
            "Vector search latency in seconds.",
            buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05),
        )
        self.pipeline_latency_hist = Histogram(
            "fr_pipeline_latency_seconds",
            "Full pipeline end-to-end latency in seconds.",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
        )

        # ── Counters ──
        self.recognition_total = Counter(
            "fr_recognition_total",
            "Total recognition attempts by result.",
            ["result"],  # recognized, unknown, ambiguous, rejected, spoof, deepfake
        )
        self.enrollment_total = Counter(
            "fr_enrollment_total",
            "Total enrollment attempts by result.",
            ["result"],  # success, quality_rejected, spoof_rejected, deepfake_rejected
        )
        self.quality_rejections = Counter(
            "fr_quality_rejections_total",
            "Quality rejections by reason.",
            ["reason"],  # blur, pose, illumination, size, composite
        )
        self.spoof_detections = Counter(
            "fr_spoof_detections_total",
            "Spoofing attempts detected.",
            ["method"],  # lbp, fft, temporal
        )
        self.deepfake_detections = Counter(
            "fr_deepfake_detections_total",
            "Deepfake attempts detected.",
            ["method"],  # frequency, compression, boundary, temporal
        )
        self.active_learning_actions = Counter(
            "fr_active_learning_actions_total",
            "Active learning actions taken.",
            ["action"],  # auto_add, verify, discard
        )

        # ── Gauges ──
        self.active_tracks = Gauge(
            "fr_active_tracks",
            "Number of currently active face tracks.",
        )
        self.index_size = Gauge(
            "fr_index_size",
            "Number of embeddings in the vector index.",
        )
        self.identity_count = Gauge(
            "fr_identity_count",
            "Number of enrolled identities.",
        )
        self.cluster_count = Gauge(
            "fr_cluster_count",
            "Total number of identity clusters.",
        )

        # ── Info ──
        self.system_info = Info(
            "fr_system",
            "System information.",
        )

    @contextmanager
    def detection_latency(self) -> Generator[None, None, None]:
        """Context manager to time detection latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.detection_latency_hist.observe(time.perf_counter() - start)

    @contextmanager
    def quality_latency(self) -> Generator[None, None, None]:
        """Context manager to time quality assessment latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.quality_latency_hist.observe(time.perf_counter() - start)

    @contextmanager
    def deepfake_latency(self) -> Generator[None, None, None]:
        """Context manager to time deepfake detection latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.deepfake_latency_hist.observe(time.perf_counter() - start)

    @contextmanager
    def antispoof_latency(self) -> Generator[None, None, None]:
        """Context manager to time anti-spoofing latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.antispoof_latency_hist.observe(time.perf_counter() - start)

    @contextmanager
    def recognition_latency(self) -> Generator[None, None, None]:
        """Context manager to time recognition latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.recognition_latency_hist.observe(time.perf_counter() - start)

    @contextmanager
    def search_latency(self) -> Generator[None, None, None]:
        """Context manager to time search latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.search_latency_hist.observe(time.perf_counter() - start)

    @contextmanager
    def pipeline_latency(self) -> Generator[None, None, None]:
        """Context manager to time full pipeline latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.pipeline_latency_hist.observe(time.perf_counter() - start)


# Module-level singleton
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Return the singleton MetricsCollector instance."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
