"""Tests for the SeverityScorer module."""

from __future__ import annotations

import pytest
from PIL import Image

from app.models.classifier import DefectType, DetectionResult
from app.models.severity import (
    SEVERITY_ACTIONS,
    SeverityLevel,
    SeverityResult,
    SeverityScorer,
)
from tests.conftest import (
    make_image_crack,
    make_image_no_defect,
    make_image_porosity,
    make_image_spatter,
    make_image_undercut,
)


def _make_detection(
    defect: DefectType, confidence: float = 0.9
) -> DetectionResult:
    return DetectionResult(defect_type=defect, confidence=confidence)


class TestSeverityLevel:
    def test_all_levels_exist(self) -> None:
        expected = {"critical", "high", "medium", "low", "none"}
        assert {l.value for l in SeverityLevel} == expected

    def test_actions_for_all_levels(self) -> None:
        for level in SeverityLevel:
            assert level in SEVERITY_ACTIONS
            assert len(SEVERITY_ACTIONS[level]) > 0


class TestSeverityResult:
    def test_is_acceptable_for_low(self) -> None:
        result = SeverityResult(
            score=10.0,
            level=SeverityLevel.LOW,
            defect_type=DefectType.SPATTER,
            confidence=0.8,
            area_fraction=0.0,
            recommended_action="PASS",
            contributing_factors=[],
        )
        assert result.is_acceptable is True

    def test_is_acceptable_for_none(self) -> None:
        result = SeverityResult(
            score=0.0,
            level=SeverityLevel.NONE,
            defect_type=DefectType.NO_DEFECT,
            confidence=0.95,
            area_fraction=0.0,
            recommended_action="PASS",
            contributing_factors=[],
        )
        assert result.is_acceptable is True

    def test_not_acceptable_for_critical(self) -> None:
        result = SeverityResult(
            score=90.0,
            level=SeverityLevel.CRITICAL,
            defect_type=DefectType.CRACK,
            confidence=0.95,
            area_fraction=0.1,
            recommended_action="REJECT",
            contributing_factors=[],
        )
        assert result.is_acceptable is False

    def test_to_dict_keys(self) -> None:
        result = SeverityResult(
            score=50.0,
            level=SeverityLevel.HIGH,
            defect_type=DefectType.POROSITY,
            confidence=0.75,
            area_fraction=0.05,
            recommended_action="action",
            contributing_factors=["factor1"],
        )
        d = result.to_dict()
        assert "score" in d
        assert "level" in d
        assert "defect_type" in d
        assert "recommended_action" in d
        assert "is_acceptable" in d


class TestSeverityScorerBasic:
    def test_no_defect_gives_none_level(self, scorer: SeverityScorer) -> None:
        detection = _make_detection(DefectType.NO_DEFECT, confidence=0.98)
        result = scorer.score(detection)
        assert result.level == SeverityLevel.NONE
        assert result.score == 0.0

    def test_crack_always_critical(self, scorer: SeverityScorer) -> None:
        """Crack must floor at CRITICAL regardless of confidence."""
        for conf in (0.3, 0.5, 0.8, 0.99):
            detection = _make_detection(DefectType.CRACK, confidence=conf)
            result = scorer.score(detection)
            assert result.level == SeverityLevel.CRITICAL, (
                f"Crack with confidence {conf} should be CRITICAL, got {result.level}"
            )
            assert result.score >= 75.0

    def test_spatter_low_severity(self, scorer: SeverityScorer) -> None:
        """Spatter is a minor defect and should score lower than cracks."""
        crack_result = scorer.score(_make_detection(DefectType.CRACK))
        spatter_result = scorer.score(_make_detection(DefectType.SPATTER))
        assert crack_result.score > spatter_result.score

    def test_incomplete_fusion_high_severity(self, scorer: SeverityScorer) -> None:
        detection = _make_detection(DefectType.INCOMPLETE_FUSION, confidence=0.9)
        result = scorer.score(detection)
        assert result.score >= 50.0

    def test_score_in_valid_range(self, scorer: SeverityScorer) -> None:
        for defect in DefectType:
            detection = _make_detection(defect, confidence=0.85)
            result = scorer.score(detection)
            assert 0.0 <= result.score <= 100.0, f"Score out of range for {defect}"

    def test_low_confidence_reduces_score(self, scorer: SeverityScorer) -> None:
        high = scorer.score(_make_detection(DefectType.POROSITY, confidence=0.95))
        low = scorer.score(_make_detection(DefectType.POROSITY, confidence=0.1))
        assert high.score > low.score

    def test_result_has_recommended_action(self, scorer: SeverityScorer) -> None:
        for defect in DefectType:
            result = scorer.score(_make_detection(defect))
            assert len(result.recommended_action) > 0

    def test_contributing_factors_list(self, scorer: SeverityScorer) -> None:
        detection = _make_detection(DefectType.CRACK)
        result = scorer.score(detection)
        assert isinstance(result.contributing_factors, list)
        # Crack always has the AWS D1.1 factor
        assert any("crack" in f.lower() or "aws" in f.lower() for f in result.contributing_factors)


class TestSeverityScorerWithImage:
    def test_score_with_image_returns_result(self, scorer: SeverityScorer) -> None:
        detection = _make_detection(DefectType.POROSITY, confidence=0.8)
        image = make_image_porosity()
        result = scorer.score(detection, image=image)
        assert isinstance(result, SeverityResult)

    def test_area_fraction_in_range(self, scorer: SeverityScorer) -> None:
        detection = _make_detection(DefectType.POROSITY, confidence=0.8)
        image = make_image_porosity()
        result = scorer.score(detection, image=image)
        assert 0.0 <= result.area_fraction <= 1.0

    def test_no_defect_with_clean_image(self, scorer: SeverityScorer) -> None:
        detection = _make_detection(DefectType.NO_DEFECT, confidence=0.97)
        image = make_image_no_defect()
        result = scorer.score(detection, image=image)
        assert result.level == SeverityLevel.NONE

    def test_crack_with_image_still_critical(self, scorer: SeverityScorer) -> None:
        detection = _make_detection(DefectType.CRACK, confidence=0.9)
        image = make_image_crack()
        result = scorer.score(detection, image=image)
        assert result.level == SeverityLevel.CRITICAL

    @pytest.mark.parametrize(
        "defect,factory",
        [
            (DefectType.POROSITY, make_image_porosity),
            (DefectType.UNDERCUT, make_image_undercut),
            (DefectType.SPATTER, make_image_spatter),
        ],
    )
    def test_score_parametrized(
        self,
        scorer: SeverityScorer,
        defect: DefectType,
        factory,
    ) -> None:
        detection = _make_detection(defect, confidence=0.8)
        image = factory()
        result = scorer.score(detection, image=image)
        assert 0.0 <= result.score <= 100.0
        assert result.level in SeverityLevel

    def test_severity_ordering(self, scorer: SeverityScorer) -> None:
        """Higher base severity defects should generally score higher at same confidence."""
        crack = scorer.score(_make_detection(DefectType.CRACK, 0.9))
        spatter = scorer.score(_make_detection(DefectType.SPATTER, 0.9))
        assert crack.score > spatter.score
