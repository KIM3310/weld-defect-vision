"""Automated inspection report generator.

Generates structured inspection reports in JSON and HTML formats,
following IIW (International Institute of Welding) documentation standards.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from jinja2 import Environment, Template

from app.models.classifier import DetectionResult
from app.models.severity import SeverityResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class InspectionReport:
    """Structured inspection report for a single weld image."""

    report_id: str
    timestamp: str
    image_filename: str
    detection: DetectionResult
    severity: SeverityResult
    preprocessing_info: dict[str, Any] = field(default_factory=dict)
    inspector_notes: str = ""
    weld_joint_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "image_filename": self.image_filename,
            "weld_joint_id": self.weld_joint_id,
            "inspector_notes": self.inspector_notes,
            "detection": {
                "defect_type": self.detection.defect_type.value,
                "confidence": round(self.detection.confidence, 4),
                "is_defect": self.detection.is_defect,
                "description": self.detection.description,
                "class_probabilities": {
                    k: round(v, 4) for k, v in self.detection.class_probabilities.items()
                },
                "demo_mode": self.detection.demo_mode,
            },
            "severity": self.severity.to_dict(),
            "preprocessing": self.preprocessing_info,
            "conclusion": self._conclusion(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def _conclusion(self) -> str:
        if not self.detection.is_defect:
            return "PASS: Weld meets quality acceptance criteria."
        level = self.severity.level.value.upper()
        dtype = self.detection.defect_type.value.replace("_", " ").title()
        return (
            f"{level}: {dtype} detected with {self.detection.confidence:.0%} confidence. "
            f"Severity score {self.severity.score:.1f}/100. "
            f"{self.severity.recommended_action}"
        )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weld Inspection Report {{ report.report_id }}</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f5f7fa; color: #222; }
  .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 8px;
               box-shadow: 0 2px 12px rgba(0,0,0,.12); overflow: hidden; }
  .header { background: #1a3a5c; color: #fff; padding: 28px 36px; }
  .header h1 { margin: 0 0 4px; font-size: 1.6em; }
  .header p  { margin: 0; opacity: .75; font-size: .95em; }
  .body { padding: 32px 36px; }
  .badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 700;
           font-size: .85em; text-transform: uppercase; letter-spacing: .5px; }
  .badge-critical { background:#fee2e2; color:#991b1b; }
  .badge-high     { background:#fef3c7; color:#92400e; }
  .badge-medium   { background:#fef9c3; color:#854d0e; }
  .badge-low      { background:#dcfce7; color:#166534; }
  .badge-none     { background:#dbeafe; color:#1e40af; }
  .badge-pass     { background:#dcfce7; color:#166534; }
  table { width: 100%; border-collapse: collapse; margin: 18px 0; }
  th { text-align: left; padding: 10px 14px; background: #f1f5f9; font-size: .82em;
       text-transform: uppercase; color: #475569; }
  td { padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-size: .95em; }
  tr:last-child td { border-bottom: none; }
  .section-title { font-size: 1.05em; font-weight: 700; color: #1a3a5c;
                   border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin: 28px 0 16px; }
  .conclusion { border-radius: 6px; padding: 16px 20px; margin-top: 24px;
                background: {{ conclusion_bg }}; border-left: 4px solid {{ conclusion_border }}; }
  .conclusion p { margin: 0; font-weight: 600; }
  .prob-bar { background: #e2e8f0; border-radius: 4px; height: 10px; width: 100%; }
  .prob-fill { background: #3b82f6; border-radius: 4px; height: 10px; }
  .footer { background: #f8fafc; padding: 16px 36px; font-size: .8em; color: #94a3b8;
            border-top: 1px solid #e2e8f0; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Weld Inspection Report</h1>
    <p>Report ID: {{ report.report_id }} &nbsp;|&nbsp; {{ report.timestamp }} &nbsp;|&nbsp;
       Generated by Weld Defect Vision AI</p>
  </div>
  <div class="body">

    <div class="section-title">Inspection Summary</div>
    <table>
      <tr><th>Image</th><td>{{ report.image_filename }}</td></tr>
      <tr><th>Weld Joint ID</th><td>{{ report.weld_joint_id or '—' }}</td></tr>
      <tr><th>Defect Type</th>
          <td><strong>{{ report.detection.defect_type.value | replace('_',' ') | title }}</strong></td></tr>
      <tr><th>Confidence</th><td>{{ "%.1f" | format(report.detection.confidence * 100) }}%</td></tr>
      <tr><th>Severity Score</th><td>{{ "%.1f" | format(report.severity.score) }} / 100</td></tr>
      <tr><th>Severity Level</th>
          <td><span class="badge badge-{{ report.severity.level.value }}">
              {{ report.severity.level.value }}</span></td></tr>
      <tr><th>Verdict</th>
          <td><span class="badge badge-{{ 'pass' if not report.detection.is_defect else report.severity.level.value }}">
              {{ 'PASS' if not report.detection.is_defect else 'ACTION REQUIRED' }}</span></td></tr>
    </table>

    <div class="section-title">Class Probability Distribution</div>
    <table>
      <tr><th>Class</th><th>Probability</th><th style="width:40%">Distribution</th></tr>
      {% for cls, prob in probs_sorted %}
      <tr>
        <td>{{ cls | replace('_',' ') | title }}</td>
        <td>{{ "%.1f" | format(prob * 100) }}%</td>
        <td>
          <div class="prob-bar">
            <div class="prob-fill" style="width:{{ "%.0f" | format(prob * 100) }}%"></div>
          </div>
        </td>
      </tr>
      {% endfor %}
    </table>

    {% if report.severity.contributing_factors %}
    <div class="section-title">Severity Contributing Factors</div>
    <ul>
      {% for factor in report.severity.contributing_factors %}
      <li>{{ factor }}</li>
      {% endfor %}
    </ul>
    {% endif %}

    <div class="section-title">Recommended Action</div>
    <p>{{ report.severity.recommended_action }}</p>

    {% if report.inspector_notes %}
    <div class="section-title">Inspector Notes</div>
    <p>{{ report.inspector_notes }}</p>
    {% endif %}

    {% if image_b64 %}
    <div class="section-title">Inspection Image</div>
    <img src="data:image/jpeg;base64,{{ image_b64 }}"
         style="max-width:100%;border-radius:6px;border:1px solid #e2e8f0;" alt="Weld image">
    {% endif %}

    <div class="conclusion">
      <p>{{ report._conclusion() }}</p>
    </div>
  </div>
  <div class="footer">
    Weld Defect Vision v0.1.0 &nbsp;|&nbsp; AI-assisted inspection — human verification required for critical findings.
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_CONCLUSION_STYLES: dict[str, tuple[str, str]] = {
    "critical": ("#fee2e2", "#ef4444"),
    "high": ("#fef3c7", "#f59e0b"),
    "medium": ("#fef9c3", "#eab308"),
    "low": ("#dcfce7", "#22c55e"),
    "none": ("#dbeafe", "#3b82f6"),
}


class ReportGenerator:
    """Generates JSON and HTML inspection reports."""

    def __init__(self) -> None:
        self._template: Template = Environment(autoescape=True).from_string(_HTML_TEMPLATE)
        self._counter: int = 0

    def generate(
        self,
        detection: DetectionResult,
        severity: SeverityResult,
        image_filename: str = "unknown",
        preprocessing_info: dict[str, Any] | None = None,
        inspector_notes: str = "",
        weld_joint_id: str = "",
        image_for_embed: bytes | None = None,
    ) -> InspectionReport:
        """Create an InspectionReport from detection and severity results."""
        self._counter += 1
        report_id = f"WDV-{datetime.now(UTC).strftime('%Y%m%d')}-{self._counter:04d}"
        timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

        return InspectionReport(
            report_id=report_id,
            timestamp=timestamp,
            image_filename=image_filename,
            detection=detection,
            severity=severity,
            preprocessing_info=preprocessing_info or {},
            inspector_notes=inspector_notes,
            weld_joint_id=weld_joint_id,
        )

    def render_html(
        self,
        report: InspectionReport,
        image_bytes: bytes | None = None,
    ) -> str:
        """Render an InspectionReport as an HTML string."""
        probs = report.detection.class_probabilities
        probs_sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

        level_key = report.severity.level.value
        bg, border = _CONCLUSION_STYLES.get(level_key, ("#f1f5f9", "#94a3b8"))

        image_b64 = ""
        if image_bytes:
            image_b64 = base64.b64encode(image_bytes).decode()

        return self._template.render(
            report=report,
            probs_sorted=probs_sorted,
            conclusion_bg=bg,
            conclusion_border=border,
            image_b64=image_b64,
        )

    def render_json(self, report: InspectionReport) -> str:
        return report.to_json()

    def generate_batch_summary(self, reports: list[InspectionReport]) -> dict[str, Any]:
        """Summarise a batch of inspection reports."""
        if not reports:
            return {"total": 0, "defects_found": 0, "pass_rate": 1.0, "by_severity": {}}

        defects = [r for r in reports if r.detection.is_defect]
        by_severity: dict[str, int] = {}
        for r in defects:
            key = r.severity.level.value
            by_severity[key] = by_severity.get(key, 0) + 1

        return {
            "total": len(reports),
            "defects_found": len(defects),
            "pass_rate": round((len(reports) - len(defects)) / len(reports), 4),
            "by_severity": by_severity,
            "defect_types": _count_by(r.detection.defect_type.value for r in defects),
        }


def _count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts
