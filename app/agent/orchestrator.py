"""Batch weld inspection agent for orchestrating inspection workflows.

Orchestrates the full weld inspection pipeline:
1. Receives batch of weld images
2. Runs preprocessing + defect classification + severity scoring
3. Prioritizes by severity (critical first)
4. Flags welds requiring immediate repair
5. Generates natural language inspection summary via OpenAI
6. Returns structured output with action items

DISCLAIMER: This is an AI-assisted tool. All findings must be confirmed by
a certified welding inspector.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "AI-assisted tool. All findings must be confirmed by certified welding inspector."
)

# Severity ordering for prioritization (higher index = higher priority)
_SEVERITY_PRIORITY: dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


class AgentStatus(StrEnum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WeldRecord:
    """Input record for a single weld image in the inspection batch."""

    image: Image.Image
    weld_joint_id: str = ""
    inspector_notes: str = ""


@dataclass
class InspectionResult:
    """Result for a single weld in the batch."""

    weld_joint_id: str
    report_id: str
    status: str  # "success" | "error"
    defect_type: str | None = None
    severity: str | None = None
    severity_score: float = 0.0
    severity_priority: int = 0
    recommended_action: str | None = None
    action_items: list[str] = field(default_factory=list)
    flagged_urgent: bool = False
    error: str | None = None


@dataclass
class InspectionSession:
    """Full output of an inspection agent run."""

    session_id: str
    status: AgentStatus
    total_welds: int
    succeeded: int
    failed: int
    urgent_cases: int
    results: list[InspectionResult] = field(default_factory=list)
    summary: str = ""
    action_items: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


def _determine_action_items(
    defect_type: str,
    severity: str,
    recommended_action: str,
) -> list[str]:
    """Determine action items based on defect type and severity level."""
    actions: list[str] = []
    severity_lower = severity.lower()
    defect_lower = defect_type.lower()

    if severity_lower == "critical":
        actions.append("URGENT: Immediate weld removal and repair required")
        actions.append("Notify QA supervisor and halt further work on joint")
    elif severity_lower == "high":
        actions.append("HOLD: Additional NDE (UT/RT) required before proceeding")
        actions.append("Submit Non-Conformance Report (NCR)")
    elif severity_lower == "medium":
        actions.append("CAUTION: Document defect location and dimensions")
        actions.append("Engineer disposition required before acceptance")
    elif severity_lower in ("low", "none"):
        actions.append("PASS: Record in inspection log for traceability")

    # Defect-specific guidance
    if defect_lower == "crack":
        actions.append("Perform MT (Magnetic Particle Testing) or PT (Penetrant Testing)")
        actions.append("Investigate root cause: preheat, interpass temperature, hydrogen")
    elif defect_lower == "porosity":
        actions.append("Review shielding gas coverage and flow rate")
        actions.append("Check electrode/wire condition and storage")
    elif defect_lower == "incomplete_fusion":
        actions.append("Review travel speed, heat input, and joint preparation")
        actions.append("Perform UT (Ultrasonic Testing) to determine depth of lack-of-fusion")
    elif defect_lower == "undercut":
        actions.append("Review welding parameters: current, voltage, travel speed")
        actions.append("Assess stress concentration per AWS D1.1 §6.9")
    elif defect_lower == "overlap":
        actions.append("Grind or blend overlap area to smooth transition")
        actions.append("Review wire feed speed and deposition rate")
    elif defect_lower == "spatter":
        actions.append("Clean spatter from heat-affected zone")
        actions.append("Review shielding gas mixture and arc parameters")

    if recommended_action:
        actions.append(f"Inspector note: {recommended_action}")

    return actions


class InspectionAgent:
    """AI agent orchestrating the full weld inspection workflow.

    Processes batches of weld images through the full analysis pipeline,
    prioritizes by severity, flags urgent cases requiring immediate repair,
    and generates a natural language summary using OpenAI.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """

    def __init__(
        self,
        api_key: str | None = None,
        components: dict[str, Any] | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise OSError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before using the InspectionAgent."
            )
        self._components = components or {}
        self._status = AgentStatus.IDLE
        self._openai_client: Any = None  # lazy-initialized

    @property
    def status(self) -> AgentStatus:
        return self._status

    def _get_openai_client(self) -> Any:
        """Lazy-initialize OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415

                self._openai_client = OpenAI(api_key=self._api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from exc
        return self._openai_client

    def _get_components(self) -> dict[str, Any]:
        """Get pipeline components, initializing defaults if needed."""
        if self._components:
            return self._components

        # Lazy-init default components
        from app.models.classifier import DefectClassifier  # noqa: PLC0415
        from app.models.severity import SeverityScorer  # noqa: PLC0415
        from app.preprocessing.pipeline import PreprocessingPipeline  # noqa: PLC0415

        self._components = {
            "classifier": DefectClassifier(demo_mode=True),
            "scorer": SeverityScorer(),
            "pipeline": PreprocessingPipeline(),
        }
        return self._components

    def _process_single(self, record: WeldRecord) -> InspectionResult:
        """Run full analysis pipeline on a single weld record."""
        report_id = str(uuid.uuid4())
        weld_joint_id = record.weld_joint_id or report_id

        components = self._get_components()
        classifier = components["classifier"]
        scorer = components["scorer"]
        pipeline = components["pipeline"]

        import io  # noqa: PLC0415

        buf = io.BytesIO()
        record.image.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        pre_result = pipeline.process_bytes(raw_bytes)
        detection = classifier.predict(pre_result.image)
        severity = scorer.score(detection, pre_result.image)

        severity_level = severity.level.value
        priority = _SEVERITY_PRIORITY.get(severity_level.lower(), 0)
        flagged = severity_level in ("critical", "high")
        action_items = _determine_action_items(
            detection.defect_type.value,
            severity_level,
            severity.recommended_action,
        )

        return InspectionResult(
            weld_joint_id=weld_joint_id,
            report_id=report_id,
            status="success",
            defect_type=detection.defect_type.value,
            severity=severity_level,
            severity_score=severity.score,
            severity_priority=priority,
            recommended_action=severity.recommended_action,
            action_items=action_items,
            flagged_urgent=flagged,
        )

    def _generate_summary(self, session: InspectionSession) -> str:
        """Generate a natural language inspection summary via OpenAI."""
        urgent_refs = [r.weld_joint_id for r in session.results if r.flagged_urgent]

        # Build defect distribution
        defect_counts: dict[str, int] = {}
        for r in session.results:
            if r.defect_type:
                defect_counts[r.defect_type] = defect_counts.get(r.defect_type, 0) + 1

        prompt = (
            f"You are a welding inspection AI assistant summarizing a batch inspection session.\n\n"
            f"Inspection session statistics:\n"
            f"- Total welds inspected: {session.total_welds}\n"
            f"- Successfully processed: {session.succeeded}\n"
            f"- Failed/errors: {session.failed}\n"
            f"- Urgent cases requiring immediate repair: {session.urgent_cases}\n"
            f"- Defect distribution: {defect_counts}\n"
            f"- Urgent weld joint IDs: {urgent_refs[:5]}\n\n"
            f"Write a concise 2-3 paragraph inspection summary for the supervising "
            f"welding engineer. Reference AWS D1.1 standards where appropriate. "
            f"Include key findings, urgent repair actions, and overall weld quality "
            f"observations. End with: '{DISCLAIMER}'"
        )

        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Failed to generate OpenAI summary: %s", exc)
            # Fallback: structured summary without AI narrative
            return (
                f"Inspection session completed. "
                f"{session.total_welds} welds processed, "
                f"{session.urgent_cases} urgent repairs identified. "
                f"Defect distribution: {defect_counts}. "
                f"{DISCLAIMER}"
            )

    def run_inspection(self, records: list[WeldRecord]) -> InspectionSession:
        """Run the full inspection workflow on a batch of weld records.

        Args:
            records: List of WeldRecord objects with images and metadata.

        Returns:
            InspectionSession with prioritized results, summary, and action items.
        """
        self._status = AgentStatus.RUNNING
        session_id = str(uuid.uuid4())

        results: list[InspectionResult] = []
        succeeded = 0
        failed = 0
        urgent_cases = 0

        for record in records:
            try:
                result = self._process_single(record)
                results.append(result)
                if result.status == "success":
                    succeeded += 1
                    if result.flagged_urgent:
                        urgent_cases += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.error(
                    "Failed to process weld %s: %s", record.weld_joint_id, exc
                )
                results.append(
                    InspectionResult(
                        weld_joint_id=record.weld_joint_id or "unknown",
                        report_id=str(uuid.uuid4()),
                        status="error",
                        error=str(exc),
                    )
                )
                failed += 1

        # Prioritize: critical/high-severity first
        results.sort(key=lambda r: r.severity_priority, reverse=True)

        # Aggregate action items
        aggregate_actions: list[str] = []
        if urgent_cases > 0:
            aggregate_actions.append(
                f"{urgent_cases} weld(s) require urgent repair — halt work on affected joints"
            )
        error_count = sum(1 for r in results if r.status == "error")
        if error_count > 0:
            aggregate_actions.append(
                f"{error_count} weld(s) could not be processed — manual inspection required"
            )

        session = InspectionSession(
            session_id=session_id,
            status=AgentStatus.COMPLETED,
            total_welds=len(records),
            succeeded=succeeded,
            failed=failed,
            urgent_cases=urgent_cases,
            results=results,
            action_items=aggregate_actions,
        )

        # Generate AI narrative summary
        session.summary = self._generate_summary(session)

        self._status = AgentStatus.COMPLETED
        return session

    def get_status(self) -> dict[str, Any]:
        """Return current agent status."""
        return {
            "status": self._status.value,
            "disclaimer": DISCLAIMER,
        }
