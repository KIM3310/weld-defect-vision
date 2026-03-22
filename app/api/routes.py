"""FastAPI route definitions for the weld defect detection API."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse
from PIL import Image

from app.models.classifier import DefectClassifier, DefectType
from app.models.severity import SeverityScorer
from app.preprocessing.pipeline import PreprocessingPipeline
from app.reporting.generator import ReportGenerator

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


class ClassesResponse(BaseModel):
    classes: list[dict[str, str]]


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
    )


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
