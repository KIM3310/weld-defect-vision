"""FastAPI application entry point for the Weld Defect Vision inference API."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import init_services, router
from app.models.classifier import DefectClassifier
from app.models.severity import SeverityScorer
from app.preprocessing.pipeline import PreprocessingPipeline
from app.reporting.generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise ML services on startup and clean up on shutdown."""
    logger.info("Initialising Weld Defect Vision services...")

    classifier = DefectClassifier(model_path=None, demo_mode=False)
    scorer = SeverityScorer()
    pipeline = PreprocessingPipeline(
        target_size=(224, 224),
        apply_clahe=True,
        apply_noise_reduction=False,
    )
    reporter = ReportGenerator()

    init_services(classifier, scorer, pipeline, reporter)

    mode = "demo" if classifier.demo_mode else "model"
    logger.info("Services ready (classifier mode: %s)", mode)

    yield

    logger.info("Shutting down Weld Defect Vision services.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Weld Defect Vision API",
    description=(
        "AI-powered welding defect detection and inspection reporting system. "
        "Developed for industrial quality assurance in shipbuilding applications."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {
        "service": "Weld Defect Vision API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
