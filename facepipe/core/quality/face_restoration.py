"""
Quality-gated face restoration module.

Applies face restoration (super-resolution, deblurring, color correction)
to low-quality face crops before recognition. Uses CodeFormer (preferred
over GFPGAN for better identity preservation) via ONNX Runtime.

Key design decisions:
  - Quality-gated: only triggered when quality score < threshold.
    Applying restoration to already-good images can HURT accuracy
    by introducing artifacts.
  - CodeFormer's controllable fidelity parameter (0.0-1.0) balances
    restoration quality vs identity preservation. 0.7-0.8 is the sweet spot.
  - A re-alignment pass after restoration is required because restoration
    can subtly shift facial geometry.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np

from facepipe.config.settings import get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class RestorationResult:
    """Result from face restoration.

    Attributes:
        restored: The restored face crop (BGR).
        was_restored: Whether restoration was actually applied.
        original_quality: Quality score of the input.
        method: Restoration method used.
        fidelity_weight: CodeFormer fidelity weight used.
    """
    restored: np.ndarray
    was_restored: bool
    original_quality: float
    method: str
    fidelity_weight: float


class FaceRestorer:
    """Quality-gated face restoration using CodeFormer.

    Only applies restoration when the input quality score falls below
    a configurable threshold. For high-quality inputs, returns the
    original image unchanged.

    Args:
        quality_threshold: Minimum quality score to skip restoration.
        fidelity_weight: CodeFormer fidelity (0.0=max restoration, 1.0=max fidelity).
        model_path: Path to CodeFormer ONNX model.
    """

    def __init__(
        self,
        quality_threshold: float = 0.45,
        fidelity_weight: float = 0.75,
        model_path: str | None = None,
    ) -> None:
        self._quality_threshold = quality_threshold
        self._fidelity_weight = fidelity_weight
        self._model_path = model_path
        self._session = None
        self._is_available = False
        self._checked = False

    def _ensure_loaded(self) -> bool:
        """Lazily load the CodeFormer ONNX model."""
        if self._checked:
            return self._is_available

        self._checked = True

        if self._model_path is None:
            default_path = Path(get_settings().models_dir) / "codeformer.onnx"
            if default_path.exists():
                self._model_path = str(default_path)

        if self._model_path is None or not Path(self._model_path).exists():
            logger.info(
                "face_restoration_unavailable",
                reason="CodeFormer ONNX model not found",
                path=self._model_path,
            )
            self._is_available = False
            return False

        try:
            import onnxruntime as ort

            providers = []
            available = ort.get_available_providers()
            for p in ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
                if p in available:
                    providers.append(p)

            self._session = ort.InferenceSession(
                self._model_path,
                providers=providers or ["CPUExecutionProvider"],
            )
            self._is_available = True
            logger.info(
                "face_restoration_loaded",
                model=self._model_path,
                fidelity=self._fidelity_weight,
            )
        except Exception as e:
            logger.warning("face_restoration_load_failed", error=str(e))
            self._is_available = False

        return self._is_available

    def restore(
        self,
        face_crop: np.ndarray,
        quality_score: float,
    ) -> RestorationResult:
        """Restore a face crop if quality is below threshold.

        Args:
            face_crop: Aligned face crop (BGR, typically 112×112).
            quality_score: Quality composite score of this face.

        Returns:
            RestorationResult with the (possibly restored) face crop.
        """
        # Quality gate: skip if quality is good enough
        if quality_score >= self._quality_threshold:
            return RestorationResult(
                restored=face_crop,
                was_restored=False,
                original_quality=quality_score,
                method="none",
                fidelity_weight=self._fidelity_weight,
            )

        # Check if model is available
        if not self._ensure_loaded():
            # Fall back to basic OpenCV enhancement
            return self._fallback_enhance(face_crop, quality_score)

        # Run CodeFormer inference
        try:
            restored = self._run_codeformer(face_crop)
            return RestorationResult(
                restored=restored,
                was_restored=True,
                original_quality=quality_score,
                method="codeformer",
                fidelity_weight=self._fidelity_weight,
            )
        except Exception as e:
            logger.warning("codeformer_inference_failed", error=str(e))
            return self._fallback_enhance(face_crop, quality_score)

    def _run_codeformer(self, face_crop: np.ndarray) -> np.ndarray:
        """Run CodeFormer ONNX inference.

        Preprocesses the input, runs the model, and post-processes
        with fidelity blending.
        """
        assert self._session is not None

        # Resize to CodeFormer's expected input size (512×512)
        original_size = face_crop.shape[:2]
        img = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Preprocess: BGR → RGB, [0,255] → [-1,1], HWC → NCHW
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_norm = (img_rgb / 255.0 - 0.5) / 0.5
        img_nchw = np.expand_dims(img_norm.transpose(2, 0, 1), axis=0)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        # Pass fidelity weight if model supports it
        inputs = {input_name: img_nchw}

        # Check if model accepts a fidelity input
        if len(self._session.get_inputs()) > 1:
            fidelity_name = self._session.get_inputs()[1].name
            fidelity_arr = np.array([self._fidelity_weight], dtype=np.float32)
            inputs[fidelity_name] = fidelity_arr

        outputs = self._session.run(None, inputs)
        output = outputs[0]

        # Post-process: NCHW → HWC, [-1,1] → [0,255], RGB → BGR
        restored = output.squeeze(0).transpose(1, 2, 0)
        restored = np.clip((restored + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

        # Blend with original based on fidelity weight
        # Higher fidelity = more original, less restoration
        original_resized = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LINEAR)
        blended = cv2.addWeighted(
            restored, 1.0 - self._fidelity_weight * 0.3,
            original_resized, self._fidelity_weight * 0.3,
            0,
        )

        # Resize back to original dimensions
        if original_size != (512, 512):
            blended = cv2.resize(blended, (original_size[1], original_size[0]))

        return blended

    def _fallback_enhance(
        self,
        face_crop: np.ndarray,
        quality_score: float,
    ) -> RestorationResult:
        """Fallback enhancement when CodeFormer is unavailable.

        Applies basic OpenCV enhancements:
        - CLAHE for contrast improvement
        - Slight sharpening via unsharp mask
        - Bilateral filtering for noise reduction
        """
        enhanced = face_crop.copy()

        # Convert to LAB for luminance-aware processing
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # CLAHE on luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l_channel)

        # Merge back
        enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Bilateral filter for noise reduction (preserves edges)
        enhanced = cv2.bilateralFilter(enhanced, 5, 40, 40)

        # Unsharp mask for slight sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)

        return RestorationResult(
            restored=enhanced,
            was_restored=True,
            original_quality=quality_score,
            method="fallback_opencv",
            fidelity_weight=self._fidelity_weight,
        )
