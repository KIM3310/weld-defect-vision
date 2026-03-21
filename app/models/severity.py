"""Defect severity scoring module.

Severity is evaluated on a 0-100 scale and mapped to ISO 5817 / AWS D1.1
quality levels:
  - Level A (Critical)   : 75-100 → weld must be rejected / repaired immediately
  - Level B (High)       : 50-74  → further NDE required
  - Level C (Medium)     : 25-49  → monitor, acceptable with documentation
  - Level D (Low)        : 0-24   → within acceptance criteria
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

from app.models.classifier import DefectType, DetectionResult


class SeverityLevel(str, Enum):
    """ISO 5817 aligned severity classification."""

    CRITICAL = "critical"   # Level A – immediate action required
    HIGH = "high"           # Level B – NDE required
    MEDIUM = "medium"       # Level C – document and monitor
    LOW = "low"             # Level D – within acceptance limits
    NONE = "none"           # No defect detected


SEVERITY_THRESHOLDS: dict[SeverityLevel, tuple[float, float]] = {
    SeverityLevel.CRITICAL: (75.0, 100.0),
    SeverityLevel.HIGH: (50.0, 74.9),
    SeverityLevel.MEDIUM: (25.0, 49.9),
    SeverityLevel.LOW: (0.0, 24.9),
    SeverityLevel.NONE: (0.0, 0.0),
}

# Base severity weights per defect type (domain knowledge from AWS D1.1)
_BASE_SEVERITY: dict[DefectType, float] = {
    DefectType.CRACK: 90.0,               # Always critical per code
    DefectType.INCOMPLETE_FUSION: 75.0,    # Structural integrity risk
    DefectType.UNDERCUT: 55.0,            # Stress concentration
    DefectType.POROSITY: 40.0,            # Depends on distribution
    DefectType.OVERLAP: 30.0,             # Surface discontinuity
    DefectType.SPATTER: 15.0,             # Cosmetic / minor
    DefectType.NO_DEFECT: 0.0,
}

SEVERITY_ACTIONS: dict[SeverityLevel, str] = {
    SeverityLevel.CRITICAL: (
        "REJECT – Immediate weld removal and repair required. "
        "Notify QA supervisor. Document with RT/UT NDE."
    ),
    SeverityLevel.HIGH: (
        "HOLD – Additional NDE (UT/RT) required before proceeding. "
        "Submit NCR (Non-Conformance Report)."
    ),
    SeverityLevel.MEDIUM: (
        "CAUTION – Document defect location and dimensions. "
        "Re-inspect after next pass. Engineer disposition required."
    ),
    SeverityLevel.LOW: (
        "PASS – Defect within acceptance criteria. "
        "Record in inspection log for traceability."
    ),
    SeverityLevel.NONE: "PASS – No defect detected. Weld meets quality standards.",
}


@dataclass
class SeverityResult:
    """Output of severity assessment."""

    score: float                  # 0-100 continuous score
    level: SeverityLevel
    defect_type: DefectType
    confidence: float
    area_fraction: float          # Estimated defect area as fraction of weld area
    recommended_action: str
    contributing_factors: list[str]

    @property
    def is_acceptable(self) -> bool:
        return self.level in (SeverityLevel.LOW, SeverityLevel.NONE)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "level": self.level.value,
            "defect_type": self.defect_type.value,
            "confidence": round(self.confidence, 4),
            "area_fraction": round(self.area_fraction, 4),
            "recommended_action": self.recommended_action,
            "contributing_factors": self.contributing_factors,
            "is_acceptable": self.is_acceptable,
        }


class SeverityScorer:
    """Scores defect severity from a DetectionResult and optionally an image.

    The score combines:
    1. Base defect type weight (domain knowledge)
    2. Classifier confidence (higher confidence → more certain)
    3. Image-based area estimation (if image provided)
    4. Spatial distribution penalty for clustered/edge defects
    """

    def score(
        self,
        detection: DetectionResult,
        image: Image.Image | None = None,
    ) -> SeverityResult:
        """Compute severity score for a detection result."""
        defect_type = detection.defect_type
        confidence = detection.confidence

        base = _BASE_SEVERITY[defect_type]
        factors: list[str] = []

        # Confidence modulation: scale base score by sqrt of confidence
        # (prevents full score when model is uncertain)
        confidence_factor = np.sqrt(confidence)
        adjusted = base * confidence_factor
        if confidence < 0.5:
            factors.append(f"Low classifier confidence ({confidence:.0%}) reduces score")
        elif confidence > 0.85:
            factors.append(f"High classifier confidence ({confidence:.0%})")

        # Image-based area and spatial analysis
        area_fraction = 0.0
        if image is not None and defect_type != DefectType.NO_DEFECT:
            area_fraction, spatial_penalty = self._analyse_image(image, defect_type)
            area_bonus = min(20.0, area_fraction * 200.0)
            adjusted = min(100.0, adjusted + area_bonus + spatial_penalty)
            if area_fraction > 0.05:
                factors.append(f"Large defect area ({area_fraction:.1%} of weld)")
            if spatial_penalty > 5:
                factors.append("Defect near weld edge (stress concentration risk)")

        # Crack always floors at CRITICAL regardless of confidence
        if defect_type == DefectType.CRACK:
            adjusted = max(adjusted, 80.0)
            factors.append("Crack – minimum CRITICAL per AWS D1.1 §6.12")

        level = self._score_to_level(adjusted)

        # Hard override: cracks are always critical
        if defect_type == DefectType.CRACK:
            level = SeverityLevel.CRITICAL

        return SeverityResult(
            score=round(adjusted, 2),
            level=level,
            defect_type=defect_type,
            confidence=confidence,
            area_fraction=area_fraction,
            recommended_action=SEVERITY_ACTIONS[level],
            contributing_factors=factors,
        )

    # ------------------------------------------------------------------
    # Image analysis helpers
    # ------------------------------------------------------------------

    def _analyse_image(
        self,
        image: Image.Image,
        defect_type: DefectType,
    ) -> tuple[float, float]:
        """Return (area_fraction, spatial_penalty)."""
        arr = np.array(image.convert("L"), dtype=np.float32)
        h, w = arr.shape

        # Threshold depends on defect type
        if defect_type in (DefectType.CRACK, DefectType.INCOMPLETE_FUSION, DefectType.UNDERCUT):
            mask = arr < 80  # dark regions
        elif defect_type in (DefectType.OVERLAP, DefectType.SPATTER):
            mask = arr > 180  # bright regions
        else:
            # Porosity: moderate dark
            mask = (arr > 30) & (arr < 100)

        area_fraction = float(mask.mean())

        # Spatial penalty: check if defect concentrated near edges
        edge_band = max(1, int(min(h, w) * 0.15))
        edge_mask = np.zeros_like(mask)
        edge_mask[:edge_band, :] = True
        edge_mask[-edge_band:, :] = True
        edge_mask[:, :edge_band] = True
        edge_mask[:, -edge_band:] = True

        if mask.sum() > 0:
            edge_defect_ratio = float((mask & edge_mask).sum() / mask.sum())
            spatial_penalty = edge_defect_ratio * 10.0
        else:
            spatial_penalty = 0.0

        return area_fraction, spatial_penalty

    @staticmethod
    def _score_to_level(score: float) -> SeverityLevel:
        if score >= 75.0:
            return SeverityLevel.CRITICAL
        if score >= 50.0:
            return SeverityLevel.HIGH
        if score >= 25.0:
            return SeverityLevel.MEDIUM
        if score > 0.0:
            return SeverityLevel.LOW
        return SeverityLevel.NONE
