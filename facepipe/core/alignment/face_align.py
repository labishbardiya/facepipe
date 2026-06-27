"""
Face alignment module.

Estimates a similarity transform from 5 detected landmarks to canonical
ArcFace/AdaFace reference positions, then applies an affine warp to
produce a 112×112 aligned face crop.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage import transform as sk_transform

from facepipe.observability.logging import get_logger

logger = get_logger(__name__)

# Canonical 5-point landmark positions for ArcFace/AdaFace (112×112 crop)
# Order: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_REFERENCE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)

# Output crop size
ALIGNED_SIZE = (112, 112)


def align_face(
    frame: np.ndarray,
    landmarks: np.ndarray,
    output_size: tuple[int, int] = ALIGNED_SIZE,
) -> np.ndarray:
    """Align a face using a similarity transform from 5-point landmarks.

    Estimates the optimal similarity transform (rotation, scale, translation)
    that maps the detected landmarks to the canonical reference positions,
    then applies it to the frame to produce an aligned crop.

    Args:
        frame: The full frame (BGR) containing the face.
        landmarks: 5 facial landmarks as shape (5, 2) in the frame coordinate system.
        output_size: Size of the output aligned crop (width, height).

    Returns:
        Aligned face crop as np.ndarray (output_size[1], output_size[0], 3), BGR.
    """
    if landmarks.shape != (5, 2):
        logger.warning("invalid_landmarks", shape=landmarks.shape)
        return cv2.resize(frame, output_size)

    # Scale reference points if output size differs from default 112×112
    reference = ARCFACE_REFERENCE.copy()
    if output_size != (112, 112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        reference[:, 0] *= scale_x
        reference[:, 1] *= scale_y

    # Estimate similarity transform (rotation + uniform scale + translation)
    tform = sk_transform.SimilarityTransform()
    tform.estimate(landmarks.astype(np.float64), reference)

    # Apply the inverse warp to the frame
    M = tform.params[:2]  # 2×3 affine matrix
    aligned = cv2.warpAffine(
        frame,
        M,
        output_size,
        borderMode=cv2.BORDER_REFLECT_101,
        flags=cv2.INTER_LINEAR,
    )

    return aligned


def align_face_batch(
    frame: np.ndarray,
    landmarks_list: list[np.ndarray],
    output_size: tuple[int, int] = ALIGNED_SIZE,
) -> list[np.ndarray]:
    """Align multiple faces from the same frame.

    Args:
        frame: The full frame (BGR).
        landmarks_list: List of 5-point landmark arrays.
        output_size: Size of each output crop.

    Returns:
        List of aligned face crops.
    """
    return [align_face(frame, lm, output_size) for lm in landmarks_list]
