"""
Adaptive decision fusion engine.

Replaces fixed similarity thresholds with a weighted composite confidence
score combining signals from every pipeline stage. Security levels shift
the decision boundary.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import numpy as np

from facepipe.config.settings import get_settings, FusionSettings, FusionSecurityLevel
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class DecisionResult:
    """Result from the decision fusion engine.

    Attributes:
        identity: The decided identity (None if not recognized).
        confidence: Composite confidence score in [0.0, 1.0].
        is_recognized: Whether the face is recognized as a known identity.
        component_scores: Per-component scores that contributed to the decision.
        decision_reason: Human-readable explanation.
        security_level: The security level used.
        active_learning_action: Recommended action (auto_add, verify, discard).
    """
    identity: Optional[str]
    confidence: float
    is_recognized: bool
    component_scores: Dict[str, float]
    decision_reason: str
    security_level: str
    active_learning_action: str


class DecisionFusionEngine:
    """Adaptive confidence scoring and decision fusion.

    Combines recognition similarity, detection confidence, face quality,
    liveness score, tracking consistency, open-set margin, and deepfake
    confidence into a single composite score.

    Args:
        settings: Fusion settings. If None, loaded from global config.
    """

    def __init__(self, settings: Optional[FusionSettings] = None) -> None:
        self._settings = settings or get_settings().fusion

    def decide(
        self,
        recognition_score: float = 0.0,
        detection_score: float = 0.0,
        quality_score: float = 0.0,
        liveness_score: float = 0.0,
        tracking_consistency: float = 0.0,
        openset_margin: float = 0.0,
        deepfake_score: float = 1.0,
        identity: Optional[str] = None,
        openset_decision: str = "unknown",
    ) -> DecisionResult:
        """Compute a fused confidence score and make a decision.

        All input scores should be in [0.0, 1.0] where higher is better.

        Args:
            recognition_score: Cosine similarity from vector search.
            detection_score: SCRFD detection confidence.
            quality_score: Face quality composite score.
            liveness_score: Anti-spoofing confidence.
            tracking_consistency: Frames tracked / re-recognition interval.
            openset_margin: Score gap between #1 and #2 match.
            deepfake_score: Deepfake realness confidence.
            identity: The candidate identity ID.
            openset_decision: Open-set decision string.

        Returns:
            DecisionResult with composite confidence and action.
        """
        s = self._settings

        # Compute weighted composite
        components = {
            "recognition": recognition_score,
            "detection": detection_score,
            "quality": quality_score,
            "liveness": liveness_score,
            "tracking": tracking_consistency,
            "openset_margin": min(openset_margin / 0.3, 1.0),  # Normalize margin
            "deepfake": deepfake_score,
        }

        weights = {
            "recognition": s.weight_recognition,
            "detection": s.weight_detection,
            "quality": s.weight_quality,
            "liveness": s.weight_liveness,
            "tracking": s.weight_tracking,
            "openset_margin": s.weight_openset_margin,
            "deepfake": s.weight_deepfake,
        }

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        composite = sum(components[k] * weights[k] for k in components)
        composite = float(np.clip(composite, 0.0, 1.0))

        # Determine threshold based on security level
        threshold_map = {
            FusionSecurityLevel.STANDARD: s.threshold_standard,
            FusionSecurityLevel.ELEVATED: s.threshold_elevated,
            FusionSecurityLevel.MAXIMUM: s.threshold_maximum,
        }
        threshold = threshold_map.get(s.security_level, s.threshold_standard)

        # Override: if open-set says ambiguous or duplicate, don't recognize
        if openset_decision in ("ambiguous", "duplicate_identity"):
            is_recognized = False
            decision_reason = f"Open-set decision: {openset_decision}. Composite={composite:.3f}."
        elif openset_decision == "unknown":
            is_recognized = False
            decision_reason = f"Unknown face. Composite={composite:.3f}, threshold={threshold:.3f}."
        elif composite >= threshold:
            is_recognized = True
            decision_reason = (
                f"Recognized as {identity} with composite={composite:.3f} "
                f"(threshold={threshold:.3f}, level={s.security_level.value})."
            )
        else:
            is_recognized = False
            decision_reason = (
                f"Below threshold: composite={composite:.3f}, "
                f"threshold={threshold:.3f}, level={s.security_level.value}."
            )

        # Active learning recommendation
        al_action = self._recommend_action(composite, is_recognized)

        return DecisionResult(
            identity=identity if is_recognized else None,
            confidence=composite,
            is_recognized=is_recognized,
            component_scores=components,
            decision_reason=decision_reason,
            security_level=s.security_level.value,
            active_learning_action=al_action,
        )

    def _recommend_action(self, confidence: float, is_recognized: bool) -> str:
        """Recommend an active learning action based on confidence."""
        from facepipe.config.settings import get_settings
        al_settings = get_settings().active_learning

        if is_recognized and confidence >= al_settings.auto_add_threshold:
            return "auto_add"
        elif confidence >= al_settings.verify_threshold:
            return "verify"
        else:
            return "discard"
