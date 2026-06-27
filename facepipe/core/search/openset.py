"""
Open-set recognition module.

Moves beyond binary known/unknown decisions to handle:
  - recognized: Clear top-1 match with sufficient margin
  - unknown: No match above recognition threshold
  - ambiguous: Top matches too close — needs verification
  - duplicate_identity: Same face matches multiple identities (enrollment error)
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from facepipe.config.settings import OpenSetSettings, get_settings
from facepipe.core.search.vector_store import SearchResult
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class MatchCandidate:
    """A candidate identity match.

    Attributes:
        identity_id: The identity ID.
        score: Similarity score.
        rank: Rank in the result list (0 = best match).
    """
    identity_id: str
    score: float
    rank: int


@dataclasses.dataclass(frozen=True)
class OpenSetResult:
    """Result from open-set recognition analysis.

    Attributes:
        decision: One of "recognized", "unknown", "ambiguous", "duplicate_identity".
        top_matches: Top-k candidate matches with scores.
        best_identity: The top identity ID (if recognized), else None.
        best_score: The top similarity score.
        margin: Score gap between #1 and #2 (0.0 if only one match).
        confidence: Calibrated confidence in the decision.
        needs_verification: Whether a human should verify this result.
        reason: Human-readable explanation of the decision.
    """
    decision: Literal["recognized", "unknown", "ambiguous", "duplicate_identity"]
    top_matches: list[MatchCandidate]
    best_identity: str | None
    best_score: float
    margin: float
    confidence: float
    needs_verification: bool
    reason: str


class OpenSetRecognizer:
    """Open-set recognition engine.

    Analyzes top-k search results to make nuanced decisions instead of
    naive "above threshold = recognized" logic.

    Args:
        settings: Open-set settings. If None, loaded from global config.
    """

    def __init__(self, settings: OpenSetSettings | None = None) -> None:
        self._settings = settings or get_settings().openset

    def analyze(self, search_results: list[SearchResult]) -> OpenSetResult:
        """Analyze search results for open-set recognition.

        Args:
            search_results: Top-k results from the vector store, sorted by score desc.

        Returns:
            OpenSetResult with decision, candidates, margin, and confidence.
        """
        if not search_results:
            return OpenSetResult(
                decision="unknown",
                top_matches=[],
                best_identity=None,
                best_score=0.0,
                margin=0.0,
                confidence=1.0,
                needs_verification=False,
                reason="No matches found in database.",
            )

        # Build candidate list
        candidates = [
            MatchCandidate(
                identity_id=r.identity_id,
                score=r.score,
                rank=i,
            )
            for i, r in enumerate(search_results)
        ]

        best = candidates[0]
        threshold = self._settings.recognition_threshold
        margin_threshold = self._settings.margin_threshold

        # Case 1: Best score below recognition threshold → unknown
        if best.score < threshold:
            return OpenSetResult(
                decision="unknown",
                top_matches=candidates,
                best_identity=None,
                best_score=best.score,
                margin=0.0,
                confidence=1.0 - best.score,
                needs_verification=False,
                reason=f"Best score {best.score:.3f} below threshold {threshold:.3f}.",
            )

        # Compute margin (only if there's a second match)
        if len(candidates) >= 2:
            # Get best score for a DIFFERENT identity than #1
            second_best = None
            for c in candidates[1:]:
                if c.identity_id != best.identity_id:
                    second_best = c
                    break

            if second_best is not None:
                margin = best.score - second_best.score
            else:
                margin = best.score  # Only one identity in results
        else:
            margin = best.score

        # Case 2: Check for duplicate identity (same face, multiple enrolled IDs)
        # Multiple different identities have similar high scores
        high_scoring = [c for c in candidates if c.score >= threshold]
        unique_identities = set(c.identity_id for c in high_scoring)

        if len(unique_identities) >= 3 and margin < margin_threshold:
            return OpenSetResult(
                decision="duplicate_identity",
                top_matches=candidates,
                best_identity=None,
                best_score=best.score,
                margin=margin,
                confidence=0.3,
                needs_verification=True,
                reason=f"Face matches {len(unique_identities)} different identities with margin {margin:.3f}. Possible enrollment error.",
            )

        # Case 3: Margin too small → ambiguous
        if margin < margin_threshold and len(unique_identities) >= 2:
            ids_str = ", ".join(c.identity_id for c in candidates[:3] if c.score >= threshold)
            return OpenSetResult(
                decision="ambiguous",
                top_matches=candidates,
                best_identity=None,
                best_score=best.score,
                margin=margin,
                confidence=0.4 + margin * 2,
                needs_verification=True,
                reason=f"Ambiguous: {ids_str} have similar scores (margin={margin:.3f}).",
            )

        # Case 4: Clear match → recognized
        confidence = min(1.0, best.score * (1.0 + margin))
        return OpenSetResult(
            decision="recognized",
            top_matches=candidates,
            best_identity=best.identity_id,
            best_score=best.score,
            margin=margin,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            needs_verification=False,
            reason=f"Recognized as {best.identity_id} with score {best.score:.3f} and margin {margin:.3f}.",
        )
