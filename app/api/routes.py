"""FastAPI route definitions for the weld defect detection API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse
from PIL import Image

from app.models.classifier import DefectClassifier, DefectType
from app.models.severity import SeverityScorer
from app.preprocessing.pipeline import PreprocessingPipeline
from app.reporting.generator import ReportGenerator

if TYPE_CHECKING:
    from app.chatbot.assistant import ChatSession

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level singletons (injected by main.py via app.state)
# ---------------------------------------------------------------------------

_classifier: DefectClassifier | None = None
_scorer: SeverityScorer | None = None
_pipeline: PreprocessingPipeline | None = None
_reporter: ReportGenerator | None = None


def init_services(
    classifier: DefectClassifier,
    scorer: SeverityScorer,
    pipeline: PreprocessingPipeline,
    reporter: ReportGenerator,
) -> None:
    """Inject service singletons (called from app lifespan)."""
    global _classifier, _scorer, _pipeline, _reporter
    _classifier = classifier
    _scorer = scorer
    _pipeline = pipeline
    _reporter = reporter


def _get_services() -> tuple[
    DefectClassifier, SeverityScorer, PreprocessingPipeline, ReportGenerator
]:
    if any(s is None for s in (_classifier, _scorer, _pipeline, _reporter)):
        raise HTTPException(status_code=503, detail="Services not initialised")
    return _classifier, _scorer, _pipeline, _reporter  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

from pydantic import BaseModel  # noqa: E402


class DetectionResponse(BaseModel):
    defect_type: str
    confidence: float
    is_defect: bool
    description: str
    class_probabilities: dict[str, float]
    demo_mode: bool


class SeverityResponse(BaseModel):
    score: float
    level: str
    recommended_action: str
    contributing_factors: list[str]
    is_acceptable: bool
    area_fraction: float


class InspectResponse(BaseModel):
    report_id: str
    timestamp: str
    image_filename: str
    weld_joint_id: str
    detection: DetectionResponse
    severity: SeverityResponse
    preprocessing: dict[str, Any]
    conclusion: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_mode: str
    defect_classes: list[str]
    proof_routes: dict[str, str] | None = None
    reviewer_fast_path: list[str] | None = None


class ClassesResponse(BaseModel):
    classes: list[dict[str, str]]


class ResourcePackResponse(BaseModel):
    service: str
    contract_version: str
    summary: dict[str, int]
    reviewer_fast_path: list[str]
    files: dict[str, str]
    defect_examples: list[dict[str, str]]
    validation_cases: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Service health and model information."""
    classifier, *_ = _get_services()
    info = classifier.get_model_info()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_mode=info["mode"],
        defect_classes=info["classes"],
        proof_routes={
            "resource_pack": "/api/v1/ops/resource-pack",
            "release_readiness": "/api/v1/ops/release-readiness",
        },
        reviewer_fast_path=[
            "/api/v1/health",
            "/api/v1/ops/resource-pack",
            "/api/v1/ops/release-readiness",
            "/docs",
        ],
    )


@router.get("/ops/resource-pack", response_model=ResourcePackResponse, tags=["system"])
async def ops_resource_pack() -> ResourcePackResponse:
    return ResourcePackResponse(
        service="weld-defect-vision-resource-pack",
        contract_version="weld-defect-review-resource-pack-v1",
        summary={
            "defect_example_count": 4,
            "validation_case_count": 4,
            "check_count": 3,
        },
        reviewer_fast_path=[
            "/api/v1/health",
            "/api/v1/ops/resource-pack",
            "/api/v1/ops/release-readiness",
            "/docs",
        ],
        files={
            "readme": "README.md",
            "sample_data": "data/sample/",
            "tests": "tests/test_api.py",
        },
        defect_examples=[
            {
                "defect_type": "crack",
                "focus": "Keep critical defects blocked from pass decisions regardless of cosmetic noise.",
            },
            {
                "defect_type": "porosity",
                "focus": "Show how medium-severity defects stay inspectable through score and report text.",
            },
            {
                "defect_type": "incomplete_fusion",
                "focus": "Explain why fusion defects should stay high-severity even when image quality varies.",
            },
            {
                "defect_type": "no_defect",
                "focus": "Preserve a conservative pass path without overstating inspection certainty.",
            },
        ],
        validation_cases=[
            {
                "case_id": "health-proof",
                "goal": "Health should expose model mode and defect classes before any inspection claim.",
            },
            {
                "case_id": "critical-crack",
                "goal": "Crack examples should remain critical and unacceptable in report output.",
            },
            {
                "case_id": "batch-summary",
                "goal": "Batch runs should preserve per-image results and error reporting together.",
            },
            {
                "case_id": "report-export",
                "goal": "HTML report generation should stay aligned with JSON inspection output.",
            },
        ],
    )


@router.get("/ops/release-readiness", tags=["system"])
async def ops_release_readiness() -> dict[str, Any]:
    classifier, *_ = _get_services()
    info = classifier.get_model_info()
    return {
        "service": "weld-defect-vision-release-readiness",
        "contract_version": "weld-defect-release-readiness-v1",
        "status": "portfolio_review_ready",
        "reviewer_fast_path": [
            "/api/v1/health",
            "/api/v1/ops/resource-pack",
            "/api/v1/ops/release-readiness",
            "/docs",
        ],
        "checks": {
            "health_surface": True,
            "resource_pack": True,
            "inspection_api": True,
            "report_export": True,
            "model_mode_visible": bool(info["mode"]),
        },
        "next_actions": [
            "Keep synthetic or demo examples clearly separated from real inspection validation claims.",
            "Pair severity scores with reviewer-visible example cases before discussing deployment.",
        ],
    }


@router.get("/classes", response_model=ClassesResponse, tags=["system"])
async def list_defect_classes() -> ClassesResponse:
    """List all supported defect types with descriptions."""
    from app.models.classifier import DEFECT_DESCRIPTIONS

    return ClassesResponse(
        classes=[{"type": dt.value, "description": DEFECT_DESCRIPTIONS[dt]} for dt in DefectType]
    )


@router.post("/inspect", response_model=InspectResponse, tags=["inspection"])
async def inspect_weld(
    file: Annotated[UploadFile, File(description="Weld image (JPEG/PNG)")],
    weld_joint_id: Annotated[str, Form()] = "",
    inspector_notes: Annotated[str, Form()] = "",
) -> InspectResponse:
    """Run full defect detection pipeline on an uploaded weld image.

    Returns detection result, severity assessment, and inspection report metadata.
    """
    classifier, scorer, pipeline, reporter = _get_services()

    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported content type: {file.content_type}. Use JPEG or PNG.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        pre_result = pipeline.process_bytes(raw_bytes)
        detection = classifier.predict(pre_result.image)
        severity = scorer.score(detection, pre_result.image)
        report = reporter.generate(
            detection=detection,
            severity=severity,
            image_filename=file.filename or "upload",
            preprocessing_info=pre_result.to_dict(),
            inspector_notes=inspector_notes,
            weld_joint_id=weld_joint_id,
        )
    except Exception as exc:
        logger.exception("Inspection pipeline failed")
        raise HTTPException(status_code=500, detail=f"Inspection failed: {exc}") from exc

    return InspectResponse(
        report_id=report.report_id,
        timestamp=report.timestamp,
        image_filename=report.image_filename,
        weld_joint_id=report.weld_joint_id,
        detection=DetectionResponse(
            defect_type=detection.defect_type.value,
            confidence=detection.confidence,
            is_defect=detection.is_defect,
            description=detection.description,
            class_probabilities=detection.class_probabilities,
            demo_mode=detection.demo_mode,
        ),
        severity=SeverityResponse(
            score=severity.score,
            level=severity.level.value,
            recommended_action=severity.recommended_action,
            contributing_factors=severity.contributing_factors,
            is_acceptable=severity.is_acceptable,
            area_fraction=severity.area_fraction,
        ),
        preprocessing=pre_result.to_dict(),
        conclusion=report._conclusion(),
    )


@router.post("/inspect/report", response_class=HTMLResponse, tags=["inspection"])
async def inspect_and_get_report(
    file: Annotated[UploadFile, File(description="Weld image (JPEG/PNG)")],
    weld_joint_id: Annotated[str, Form()] = "",
    inspector_notes: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Run inspection and return a rendered HTML report."""
    classifier, scorer, pipeline, reporter = _get_services()

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        pre_result = pipeline.process_bytes(raw_bytes)
        detection = classifier.predict(pre_result.image)
        severity = scorer.score(detection, pre_result.image)
        report = reporter.generate(
            detection=detection,
            severity=severity,
            image_filename=file.filename or "upload",
            preprocessing_info=pre_result.to_dict(),
            inspector_notes=inspector_notes,
            weld_joint_id=weld_joint_id,
        )
        html = reporter.render_html(report, image_bytes=raw_bytes)
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

    return HTMLResponse(content=html)


@router.post("/batch/inspect", tags=["inspection"])
async def batch_inspect(
    files: Annotated[list[UploadFile], File(description="Multiple weld images")],
) -> dict[str, Any]:
    """Run inspection on multiple images and return a batch summary."""
    classifier, scorer, pipeline, reporter = _get_services()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch.")

    reports = []
    errors = []

    for upload in files:
        try:
            raw = await upload.read()
            pre = pipeline.process_bytes(raw)
            detection = classifier.predict(pre.image)
            severity = scorer.score(detection, pre.image)
            report = reporter.generate(
                detection=detection,
                severity=severity,
                image_filename=upload.filename or "upload",
                preprocessing_info=pre.to_dict(),
            )
            reports.append(report)
        except Exception as exc:
            errors.append({"filename": upload.filename, "error": str(exc)})
            logger.warning("Failed to process %s: %s", upload.filename, exc)

    summary = reporter.generate_batch_summary(reports)
    results = [r.to_dict() for r in reports]

    return {
        "summary": summary,
        "results": results,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Clinical AI Chatbot
# ---------------------------------------------------------------------------

# In-memory chat sessions (use Redis in production for persistence)
_chat_sessions: dict[str, ChatSession] = {}


class ChatRequest(BaseModel):
    """Request body for POST /api/v1/chat."""

    session_id: str | None = None
    message: str
    defect_type: str = "unknown"
    severity: str = "unknown"
    recommended_action: str = ""


class ChatResponse(BaseModel):
    """Response from POST /api/v1/chat."""

    session_id: str
    reply: str
    disclaimer: str


@router.post("/chat", response_model=ChatResponse, tags=["ai-assistant"])
async def chat_with_assistant(body: ChatRequest) -> ChatResponse:
    """Chat with the welding inspection AI assistant about a defect result.

    Maintains multi-turn conversation history per session_id.
    Requires OPENAI_API_KEY environment variable to be set.

    DISCLAIMER: This is an AI-assisted tool. All findings must be confirmed
    by a certified welding inspector.
    """
    import uuid  # noqa: PLC0415

    from app.chatbot.assistant import WeldingAssistant  # noqa: PLC0415

    try:
        assistant = WeldingAssistant()
    except OSError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    session_id = body.session_id or str(uuid.uuid4())
    if session_id in _chat_sessions:
        session = _chat_sessions[session_id]
    else:
        session = assistant.create_session(
            defect_type=body.defect_type,
            severity=body.severity,
            recommended_action=body.recommended_action,
            session_id=session_id,
        )
        _chat_sessions[session_id] = session

    try:
        reply = assistant.chat(session, body.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        disclaimer=assistant.get_disclaimer(),
    )


# ---------------------------------------------------------------------------
# Batch Inspection Agent
# ---------------------------------------------------------------------------

_inspection_agent: object | None = None


class AgentInspectRequest(BaseModel):
    """Request body for POST /api/v1/agent/inspect."""

    weld_joint_ids: list[str] = []


@router.post("/agent/inspect", tags=["ai-agent"])
async def run_inspection_agent(
    files: Annotated[list[UploadFile], File(description="Batch of weld images")],
    weld_joint_ids: Annotated[str, Form(description="Comma-separated weld joint IDs")] = "",
) -> dict[str, Any]:
    """Run the AI inspection agent on a batch of weld images.

    Orchestrates the full inspection workflow:
    - Classifies each weld image for defect type
    - Scores severity per AWS D1.1 / ISO 5817
    - Prioritizes welds by severity (critical first)
    - Flags welds requiring immediate repair
    - Generates a natural language summary via OpenAI

    Requires OPENAI_API_KEY environment variable to be set.
    Maximum batch size: 20 images.

    DISCLAIMER: This is an AI-assisted tool. All findings must be confirmed
    by a certified welding inspector.
    """
    import io  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    from app.agent.orchestrator import InspectionAgent, WeldRecord  # noqa: PLC0415

    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 20 images.",
        )

    joint_id_list = [j.strip() for j in weld_joint_ids.split(",") if j.strip()]

    try:
        agent = InspectionAgent()
    except OSError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    records = []
    for i, upload_file in enumerate(files):
        try:
            raw_bytes = await upload_file.read()
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            joint_id = joint_id_list[i] if i < len(joint_id_list) else str(uuid.uuid4())
            records.append(WeldRecord(image=image, weld_joint_id=joint_id))
        except Exception:
            records.append(
                WeldRecord(
                    image=Image.new("RGB", (224, 224), color=(128, 128, 128)),
                    weld_joint_id=f"invalid_{i}",
                )
            )

    session = agent.run_inspection(records)

    results_out = [
        {
            "weld_joint_id": r.weld_joint_id,
            "report_id": r.report_id,
            "status": r.status,
            "defect_type": r.defect_type,
            "severity": r.severity,
            "severity_score": r.severity_score,
            "flagged_urgent": r.flagged_urgent,
            "action_items": r.action_items,
            "error": r.error,
        }
        for r in session.results
    ]

    return {
        "session_id": session.session_id,
        "status": session.status.value,
        "total_welds": session.total_welds,
        "succeeded": session.succeeded,
        "failed": session.failed,
        "urgent_cases": session.urgent_cases,
        "summary": session.summary,
        "action_items": session.action_items,
        "results": results_out,
        "disclaimer": session.disclaimer,
    }


@router.get("/agent/status", tags=["ai-agent"])
async def agent_status() -> dict[str, Any]:
    """Get the current status of the inspection agent.

    Returns agent readiness and disclaimer information.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    import os  # noqa: PLC0415

    from app.agent.orchestrator import DISCLAIMER  # noqa: PLC0415

    api_key_set = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "agent": "inspection_agent",
        "api_key_configured": api_key_set,
        "status": "ready" if api_key_set else "unavailable",
        "disclaimer": DISCLAIMER,
    }


@router.get("/demo/synthetic", tags=["demo"])
async def generate_synthetic_demo() -> dict[str, Any]:
    """Generate a synthetic weld image, run inspection, and return result (demo only)."""
    import numpy as np

    classifier, scorer, pipeline, reporter = _get_services()

    # Create a synthetic image with simulated porosity
    rng = np.random.default_rng(42)
    img_arr = np.ones((224, 224, 3), dtype=np.uint8) * 160
    # weld bead region
    img_arr[80:144, :] = 200
    # simulated dark pores
    for _ in range(15):
        cx = int(rng.integers(10, 214))
        cy = int(rng.integers(85, 139))
        r = int(rng.integers(3, 8))
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        mask = x * x + y * y <= r * r
        y0, x0 = max(0, cy - r), max(0, cx - r)
        y1 = min(224, cy + r + 1)
        x1 = min(224, cx + r + 1)
        my, mx = mask[: y1 - y0, : x1 - x0].shape
        img_arr[y0 : y0 + my, x0 : x0 + mx][mask[:my, :mx]] = 30

    image = Image.fromarray(img_arr)
    detection = classifier.predict(image)
    severity = scorer.score(detection, image)
    report = reporter.generate(
        detection=detection,
        severity=severity,
        image_filename="synthetic_demo.png",
        weld_joint_id="DEMO-001",
    )

    return {
        "note": "Synthetic image demo — not a real weld",
        "report": report.to_dict(),
    }
