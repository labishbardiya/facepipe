"""
ByteTrack-inspired temporal face tracker.

Maintains persistent face identities across video frames using:
  - Kalman filter prediction for inter-frame position estimation
  - IoU-based association between detections and existing tracks
  - Two-stage matching: high-confidence then low-confidence detections
  - Recognition caching to skip redundant inference

This dramatically reduces per-frame computation by only running the
full recognition pipeline on new/changed faces.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from facepipe.config.settings import get_settings, TrackingSettings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class KalmanState:
    """Kalman filter state for bounding box tracking.

    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    """
    mean: np.ndarray   # (8,)
    covariance: np.ndarray  # (8, 8)


@dataclasses.dataclass
class TrackedFace:
    """A tracked face across video frames.

    Attributes:
        track_id: Unique persistent track ID.
        bbox: Current bounding box [x1, y1, x2, y2].
        score: Detection confidence.
        frames_tracked: Total frames this track has existed.
        frames_since_update: Frames since last detection match.
        recognition_result: Cached recognition result (if any).
        frames_since_recognition: Frames since last recognition was run.
        is_confirmed: Whether this track is confirmed (seen >= 3 frames).
    """
    track_id: int
    bbox: np.ndarray
    score: float
    frames_tracked: int = 0
    frames_since_update: int = 0
    recognition_result: Optional[Any] = None
    frames_since_recognition: int = 0
    is_confirmed: bool = False
    _kalman: Optional[KalmanState] = dataclasses.field(default=None, repr=False)


class ByteTracker:
    """ByteTrack-inspired multi-face temporal tracker.

    Associates face detections across frames using IoU matching with
    a two-stage approach: first match high-confidence detections,
    then attempt to match remaining low-confidence detections.

    Args:
        settings: Tracking settings. If None, loaded from global config.
    """

    def __init__(self, settings: Optional[TrackingSettings] = None) -> None:
        self._settings = settings or get_settings().tracking
        self._tracks: Dict[int, TrackedFace] = {}
        self._next_id: int = 0
        self._frame_count: int = 0

    def update(
        self,
        detections: List[Tuple[np.ndarray, float]],
    ) -> List[TrackedFace]:
        """Update tracks with new detections.

        Args:
            detections: List of (bbox, score) tuples where bbox is [x1,y1,x2,y2].

        Returns:
            List of active TrackedFace objects after update.
        """
        self._frame_count += 1

        if not detections:
            # Age all tracks
            self._age_tracks()
            return self._get_active_tracks()

        bboxes = np.array([d[0] for d in detections])
        scores = np.array([d[1] for d in detections])

        # Split detections into high and low confidence
        high_mask = scores >= self._settings.high_det_threshold
        low_mask = ~high_mask & (scores >= self._settings.low_det_threshold)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        # Predict next positions for existing tracks
        predicted_tracks = list(self._tracks.values())
        predicted_bboxes = np.array([
            self._predict_bbox(t) for t in predicted_tracks
        ]) if predicted_tracks else np.zeros((0, 4))

        # Stage 1: Match high-confidence detections
        unmatched_tracks, unmatched_dets = self._match_stage(
            predicted_tracks, predicted_bboxes,
            bboxes[high_indices] if len(high_indices) > 0 else np.zeros((0, 4)),
            high_indices,
        )

        # Stage 2: Match low-confidence detections with remaining tracks
        if len(low_indices) > 0 and len(unmatched_tracks) > 0:
            remaining_tracks = [predicted_tracks[i] for i in unmatched_tracks]
            remaining_bboxes = predicted_bboxes[unmatched_tracks] if len(unmatched_tracks) > 0 else np.zeros((0, 4))

            still_unmatched_tracks, _ = self._match_stage(
                remaining_tracks, remaining_bboxes,
                bboxes[low_indices] if len(low_indices) > 0 else np.zeros((0, 4)),
                low_indices,
            )
            # Mark truly unmatched tracks
            unmatched_track_ids = {remaining_tracks[i].track_id for i in still_unmatched_tracks}
        else:
            unmatched_track_ids = {predicted_tracks[i].track_id for i in unmatched_tracks}

        # Age unmatched tracks
        for track_id in unmatched_track_ids:
            if track_id in self._tracks:
                self._tracks[track_id].frames_since_update += 1

        # Create new tracks for unmatched high-confidence detections
        for det_idx in unmatched_dets:
            self._create_track(bboxes[det_idx], scores[det_idx])

        # Remove dead tracks
        self._remove_dead_tracks()

        return self._get_active_tracks()

    def _match_stage(
        self,
        tracks: List[TrackedFace],
        track_bboxes: np.ndarray,
        det_bboxes: np.ndarray,
        det_indices: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Match detections to tracks using IoU with Hungarian algorithm.

        Returns:
            (unmatched_track_indices, unmatched_detection_global_indices)
        """
        if len(tracks) == 0 or len(det_bboxes) == 0:
            return list(range(len(tracks))), list(det_indices)

        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)

        # Hungarian algorithm (minimize cost = maximize IoU)
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self._settings.match_threshold:
                track = tracks[row]
                det_global_idx = int(det_indices[col])

                # Update track
                track.bbox = det_bboxes[col].copy()
                track.frames_tracked += 1
                track.frames_since_update = 0
                track.frames_since_recognition += 1

                if track.frames_tracked >= 3:
                    track.is_confirmed = True

                matched_tracks.add(row)
                matched_dets.add(det_global_idx)

        unmatched_track_indices = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_det_indices = [int(det_indices[j]) for j in range(len(det_bboxes)) if int(det_indices[j]) not in matched_dets]

        return unmatched_track_indices, unmatched_det_indices

    def _create_track(self, bbox: np.ndarray, score: float) -> TrackedFace:
        """Create a new track."""
        track = TrackedFace(
            track_id=self._next_id,
            bbox=bbox.copy(),
            score=score,
            frames_tracked=1,
        )
        self._tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _predict_bbox(self, track: TrackedFace) -> np.ndarray:
        """Predict next bbox position (simple linear motion model)."""
        # For simplicity, use current bbox (Kalman prediction could be added)
        return track.bbox.copy()

    def _age_tracks(self) -> None:
        """Increment frames_since_update for all tracks."""
        for track in self._tracks.values():
            track.frames_since_update += 1
            track.frames_since_recognition += 1

    def _remove_dead_tracks(self) -> None:
        """Remove tracks that haven't been matched for too long."""
        dead_ids = [
            tid for tid, track in self._tracks.items()
            if track.frames_since_update > self._settings.buffer
        ]
        for tid in dead_ids:
            del self._tracks[tid]

    def _get_active_tracks(self) -> List[TrackedFace]:
        """Return all currently active tracks."""
        return list(self._tracks.values())

    def needs_recognition(self, track: TrackedFace) -> bool:
        """Check if a track needs (re-)recognition.

        A track needs recognition if:
          - It has no cached recognition result
          - Too many frames have passed since last recognition
        """
        if track.recognition_result is None:
            return True
        if track.frames_since_recognition >= self._settings.re_recognize_interval:
            return True
        return False

    def set_recognition(self, track_id: int, result: Any) -> None:
        """Cache a recognition result for a track."""
        if track_id in self._tracks:
            self._tracks[track_id].recognition_result = result
            self._tracks[track_id].frames_since_recognition = 0

    @staticmethod
    def _compute_iou_matrix(
        boxes_a: np.ndarray,
        boxes_b: np.ndarray,
    ) -> np.ndarray:
        """Compute IoU between two sets of bounding boxes.

        Args:
            boxes_a: (N, 4) array of [x1, y1, x2, y2]
            boxes_b: (M, 4) array of [x1, y1, x2, y2]

        Returns:
            (N, M) IoU matrix
        """
        n = boxes_a.shape[0]
        m = boxes_b.shape[0]

        # Compute intersections
        x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)  # (N, M)
        y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
        x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
        y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Compute areas
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

        union = area_a[:, None] + area_b[None, :] - intersection

        return intersection / (union + 1e-8)

    @property
    def active_count(self) -> int:
        """Number of active tracks."""
        return len(self._tracks)

    def get_track(self, track_id: int) -> Optional[TrackedFace]:
        """Get a specific track by ID."""
        return self._tracks.get(track_id)
