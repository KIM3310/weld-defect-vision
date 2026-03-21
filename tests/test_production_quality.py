"""Production quality and integration tests.

Validates end-to-end pipeline behaviour, edge cases, and system contracts
that would matter in a real shipbuilding inspection environment.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from app.models.classifier import DefectClassifier, DefectType, DetectionResult
from app.models.severity import SeverityScorer
from app.preprocessing.pipeline import PreprocessingPipeline
from app.reporting.generator import ReportGenerator
from tests.conftest import (
    make_image_crack,
    make_image_no_defect,
    make_image_porosity,
    make_image_spatter,
    make_image_undercut,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_pipeline() -> tuple[
    DefectClassifier, PreprocessingPipeline, SeverityScorer, ReportGenerator
]:
    clf = DefectClassifier(demo_mode=True)
    pipe = PreprocessingPipeline(target_size=(224, 224), apply_clahe=True)
    scorer = SeverityScorer()
    reporter = ReportGenerator()
    return clf, pipe, scorer, reporter


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    def _run(
        self,
        full_pipeline: tuple,
        image: Image.Image,
        joint_id: str = "TEST-001",
    ) -> dict:
        clf, pipe, scorer, reporter = full_pipeline
        pre = pipe.process(image)
        detection = clf.predict(pre.image)
        severity = scorer.score(detection, pre.image)
        report = reporter.generate(
            detection=detection,
            severity=severity,
            image_filename="test.png",
            weld_joint_id=joint_id,
            preprocessing_info=pre.to_dict(),
        )
        return report.to_dict()

    def test_full_pipeline_no_defect(self, full_pipeline: tuple) -> None:
        result = self._run(full_pipeline, make_image_no_defect())
        assert "report_id" in result
        assert result["detection"]["confidence"] >= 0.0

    def test_full_pipeline_porosity(self, full_pipeline: tuple) -> None:
        result = self._run(full_pipeline, make_image_porosity())
        assert "severity" in result
        assert 0.0 <= result["severity"]["score"] <= 100.0

    def test_full_pipeline_crack(self, full_pipeline: tuple) -> None:
        result = self._run(full_pipeline, make_image_crack())
        # Crack must always be CRITICAL
        assert result["severity"]["level"] == "critical"

    def test_full_pipeline_report_json_valid(self, full_pipeline: tuple) -> None:
        clf, pipe, scorer, reporter = full_pipeline
        pre = pipe.process(make_image_no_defect())
        detection = clf.predict(pre.image)
        severity = scorer.score(detection)
        report = reporter.generate(detection=detection, severity=severity)
        json_str = reporter.render_json(report)
        parsed = json.loads(json_str)
        assert parsed["report_id"].startswith("WDV-")

    def test_full_pipeline_html_report_valid(self, full_pipeline: tuple) -> None:
        clf, pipe, scorer, reporter = full_pipeline
        pre = pipe.process(make_image_crack())
        detection = clf.predict(pre.image)
        severity = scorer.score(detection, pre.image)
        report = reporter.generate(detection=detection, severity=severity)
        html = reporter.render_html(report)
        assert "<!DOCTYPE html>" in html
        assert "Weld Inspection Report" in html

    @pytest.mark.parametrize(
        "factory",
        [
            make_image_no_defect,
            make_image_porosity,
            make_image_crack,
            make_image_undercut,
            make_image_spatter,
        ],
    )
    def test_pipeline_all_defect_types(self, full_pipeline: tuple, factory) -> None:
        result = self._run(full_pipeline, factory())
        assert result["severity"]["score"] >= 0.0
        assert result["detection"]["defect_type"] in {d.value for d in DefectType}


# ---------------------------------------------------------------------------
# Data contract / schema tests
# ---------------------------------------------------------------------------


class TestDataContracts:
    def test_detection_result_serialisable(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        result = clf.predict(make_image_no_defect())
        d = {
            "defect_type": result.defect_type.value,
            "confidence": result.confidence,
            "is_defect": result.is_defect,
            "class_probabilities": result.class_probabilities,
        }
        assert json.dumps(d)  # must be JSON-serialisable

    def test_severity_result_serialisable(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        scorer = SeverityScorer()
        detection = clf.predict(make_image_porosity())
        severity = scorer.score(detection, make_image_porosity())
        d = severity.to_dict()
        assert json.dumps(d)

    def test_report_to_dict_fully_serialisable(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        scorer = SeverityScorer()
        reporter = ReportGenerator()
        detection = clf.predict(make_image_no_defect())
        severity = scorer.score(detection)
        report = reporter.generate(detection=detection, severity=severity)
        assert json.dumps(report.to_dict())

    def test_class_probabilities_sum_to_one(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        for factory in (make_image_no_defect, make_image_porosity, make_image_crack):
            result = clf.predict(factory())
            total = sum(result.class_probabilities.values())
            assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}"

    def test_all_defect_types_covered_in_probs(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        result = clf.predict(make_image_no_defect())
        for defect in DefectType:
            assert defect.value in result.class_probabilities


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_tiny_image(self) -> None:
        """1x1 pixel image should not crash the pipeline."""
        tiny = Image.fromarray(np.array([[[128, 128, 128]]], dtype=np.uint8))
        pipe = PreprocessingPipeline()
        clf = DefectClassifier(demo_mode=True)
        pre = pipe.process(tiny)
        result = clf.predict(pre.image)
        assert isinstance(result, DetectionResult)

    def test_all_black_image(self) -> None:
        black = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        clf = DefectClassifier(demo_mode=True)
        result = clf.predict(black)
        assert 0.0 <= result.confidence <= 1.0

    def test_all_white_image(self) -> None:
        white = Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8))
        clf = DefectClassifier(demo_mode=True)
        result = clf.predict(white)
        assert 0.0 <= result.confidence <= 1.0

    def test_rgba_image_handled(self) -> None:
        rgba = Image.fromarray(np.full((224, 224, 4), 200, dtype=np.uint8), mode="RGBA")
        pipe = PreprocessingPipeline()
        result = pipe.process(rgba)
        assert result.image.mode == "RGB"

    def test_high_noise_image(self) -> None:
        rng = np.random.default_rng(99)
        noisy = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
        clf = DefectClassifier(demo_mode=True)
        result = clf.predict(noisy)
        assert isinstance(result, DetectionResult)

    def test_batch_predict_empty_list(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        results = clf.predict_batch([])
        assert results == []

    def test_batch_summary_empty(self) -> None:
        reporter = ReportGenerator()
        summary = reporter.generate_batch_summary([])
        assert summary["total"] == 0
        assert summary["pass_rate"] == 1.0

    def test_severity_score_no_image(self) -> None:
        scorer = SeverityScorer()
        clf = DefectClassifier(demo_mode=True)
        detection = clf.predict(make_image_porosity())
        result = scorer.score(detection, image=None)
        assert 0.0 <= result.score <= 100.0

    def test_preprocessing_large_image(self) -> None:
        large = Image.fromarray(np.ones((2000, 3000, 3), dtype=np.uint8) * 180)
        pipe = PreprocessingPipeline(target_size=(224, 224))
        result = pipe.process(large)
        assert result.processed_size == (224, 224)


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_image_same_result(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        img = make_image_porosity()
        r1 = clf.predict(img)
        r2 = clf.predict(img)
        assert r1.defect_type == r2.defect_type
        assert abs(r1.confidence - r2.confidence) < 1e-6

    def test_preprocessing_deterministic(self) -> None:
        pipe = PreprocessingPipeline()
        img = make_image_crack()
        r1 = pipe.process(img)
        r2 = pipe.process(img)
        arr1 = np.array(r1.image)
        arr2 = np.array(r2.image)
        assert np.array_equal(arr1, arr2)
