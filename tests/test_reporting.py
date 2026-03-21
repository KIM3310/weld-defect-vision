"""Tests for the inspection report generator."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from app.models.classifier import DefectType, DetectionResult
from app.models.severity import SeverityLevel, SeverityResult
from app.reporting.generator import InspectionReport, ReportGenerator
from tests.conftest import make_image_no_defect, make_image_porosity, image_to_bytes


def _make_detection(defect: DefectType = DefectType.POROSITY, confidence: float = 0.82) -> DetectionResult:
    probs = {d.value: 0.0 for d in DefectType}
    probs[defect.value] = confidence
    probs[DefectType.NO_DEFECT.value] = 1.0 - confidence
    return DetectionResult(
        defect_type=defect,
        confidence=confidence,
        class_probabilities=probs,
    )


def _make_severity(
    defect: DefectType = DefectType.POROSITY,
    score: float = 45.0,
    level: SeverityLevel = SeverityLevel.MEDIUM,
) -> SeverityResult:
    return SeverityResult(
        score=score,
        level=level,
        defect_type=defect,
        confidence=0.82,
        area_fraction=0.03,
        recommended_action="Monitor and document.",
        contributing_factors=["Moderate area"],
    )


class TestReportGeneration:
    def test_generate_returns_inspection_report(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(
            detection=_make_detection(),
            severity=_make_severity(),
            image_filename="test.png",
        )
        assert isinstance(report, InspectionReport)

    def test_report_id_format(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(
            detection=_make_detection(),
            severity=_make_severity(),
        )
        assert report.report_id.startswith("WDV-")

    def test_report_timestamp_set(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(
            detection=_make_detection(),
            severity=_make_severity(),
        )
        assert report.timestamp.endswith("Z")
        assert "T" in report.timestamp

    def test_report_id_increments(self, reporter: ReportGenerator) -> None:
        r1 = reporter.generate(detection=_make_detection(), severity=_make_severity())
        r2 = reporter.generate(detection=_make_detection(), severity=_make_severity())
        assert r1.report_id != r2.report_id

    def test_image_filename_stored(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(
            detection=_make_detection(),
            severity=_make_severity(),
            image_filename="weld_bead_42.png",
        )
        assert report.image_filename == "weld_bead_42.png"

    def test_weld_joint_id_stored(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(
            detection=_make_detection(),
            severity=_make_severity(),
            weld_joint_id="J-2024-007",
        )
        assert report.weld_joint_id == "J-2024-007"

    def test_inspector_notes_stored(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(
            detection=_make_detection(),
            severity=_make_severity(),
            inspector_notes="Surface visually inspected.",
        )
        assert report.inspector_notes == "Surface visually inspected."

    def test_preprocessing_info_stored(self, reporter: ReportGenerator) -> None:
        info = {"clahe_applied": True, "steps_applied": ["resize"]}
        report = reporter.generate(
            detection=_make_detection(),
            severity=_make_severity(),
            preprocessing_info=info,
        )
        assert report.preprocessing_info == info


class TestReportToDict:
    def test_to_dict_has_required_keys(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(
            detection=_make_detection(),
            severity=_make_severity(),
        )
        d = report.to_dict()
        for key in ("report_id", "timestamp", "detection", "severity", "conclusion"):
            assert key in d

    def test_to_dict_detection_structure(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(detection=_make_detection(), severity=_make_severity())
        d = report.to_dict()
        detection = d["detection"]
        assert "defect_type" in detection
        assert "confidence" in detection
        assert "is_defect" in detection

    def test_conclusion_pass_for_no_defect(self, reporter: ReportGenerator) -> None:
        detection = _make_detection(DefectType.NO_DEFECT, 0.97)
        severity = _make_severity(DefectType.NO_DEFECT, 0.0, SeverityLevel.NONE)
        report = reporter.generate(detection=detection, severity=severity)
        assert "PASS" in report._conclusion()

    def test_conclusion_action_required_for_defect(self, reporter: ReportGenerator) -> None:
        detection = _make_detection(DefectType.CRACK, 0.95)
        severity = _make_severity(DefectType.CRACK, 90.0, SeverityLevel.CRITICAL)
        report = reporter.generate(detection=detection, severity=severity)
        conclusion = report._conclusion()
        assert "CRITICAL" in conclusion or "crack" in conclusion.lower()


class TestRenderJson:
    def test_render_json_valid_json(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(detection=_make_detection(), severity=_make_severity())
        json_str = reporter.render_json(report)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_render_json_has_report_id(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(detection=_make_detection(), severity=_make_severity())
        parsed = json.loads(reporter.render_json(report))
        assert parsed["report_id"].startswith("WDV-")


class TestRenderHtml:
    def test_render_html_returns_string(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(detection=_make_detection(), severity=_make_severity())
        html = reporter.render_html(report)
        assert isinstance(html, str)

    def test_render_html_valid_document(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(detection=_make_detection(), severity=_make_severity())
        html = reporter.render_html(report)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_render_html_contains_report_id(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(detection=_make_detection(), severity=_make_severity())
        html = reporter.render_html(report)
        assert report.report_id in html

    def test_render_html_with_image_bytes(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(detection=_make_detection(), severity=_make_severity())
        image_bytes = image_to_bytes(make_image_no_defect())
        html = reporter.render_html(report, image_bytes=image_bytes)
        assert "data:image/jpeg;base64," in html

    def test_render_html_without_image(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(detection=_make_detection(), severity=_make_severity())
        html = reporter.render_html(report, image_bytes=None)
        assert "<!DOCTYPE html>" in html

    def test_render_html_severity_level_present(self, reporter: ReportGenerator) -> None:
        report = reporter.generate(
            detection=_make_detection(DefectType.CRACK, 0.95),
            severity=_make_severity(DefectType.CRACK, 90.0, SeverityLevel.CRITICAL),
        )
        html = reporter.render_html(report)
        assert "critical" in html.lower()


class TestBatchSummary:
    def _make_report(
        self,
        reporter: ReportGenerator,
        defect: DefectType,
        level: SeverityLevel,
        score: float,
    ) -> InspectionReport:
        return reporter.generate(
            detection=_make_detection(defect),
            severity=_make_severity(defect, score, level),
        )

    def test_empty_batch_summary(self, reporter: ReportGenerator) -> None:
        summary = reporter.generate_batch_summary([])
        assert summary["total"] == 0
        assert summary["pass_rate"] == 1.0

    def test_all_pass_summary(self, reporter: ReportGenerator) -> None:
        reports = [
            self._make_report(reporter, DefectType.NO_DEFECT, SeverityLevel.NONE, 0.0)
            for _ in range(5)
        ]
        summary = reporter.generate_batch_summary(reports)
        assert summary["total"] == 5
        assert summary["defects_found"] == 0
        assert summary["pass_rate"] == 1.0

    def test_mixed_batch_summary(self, reporter: ReportGenerator) -> None:
        reports = [
            self._make_report(reporter, DefectType.NO_DEFECT, SeverityLevel.NONE, 0.0),
            self._make_report(reporter, DefectType.CRACK, SeverityLevel.CRITICAL, 90.0),
            self._make_report(reporter, DefectType.POROSITY, SeverityLevel.MEDIUM, 40.0),
        ]
        summary = reporter.generate_batch_summary(reports)
        assert summary["total"] == 3
        assert summary["defects_found"] == 2
        assert abs(summary["pass_rate"] - 1 / 3) < 0.01

    def test_by_severity_counted(self, reporter: ReportGenerator) -> None:
        reports = [
            self._make_report(reporter, DefectType.CRACK, SeverityLevel.CRITICAL, 90.0),
            self._make_report(reporter, DefectType.CRACK, SeverityLevel.CRITICAL, 88.0),
        ]
        summary = reporter.generate_batch_summary(reports)
        assert summary["by_severity"].get("critical", 0) == 2
