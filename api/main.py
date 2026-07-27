"""FastAPI inference server for weld defect detection."""

import io
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from src.config import DEFECT_LABELS
from src.inference import WeldDefectDetector
from src.visualize import draw_detections

app = FastAPI(
    title="Weld Defect Vision",
    description="Weld defect detection API using YOLOv8",
    version="0.1.0",
)

CHECKPOINT_PATH = Path("checkpoints/best.pt")
detector: WeldDefectDetector | None = None


def get_detector() -> WeldDefectDetector:
    global detector
    if detector is None:
        if not CHECKPOINT_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model checkpoint not found. Train the model first.",
            )
        detector = WeldDefectDetector(CHECKPOINT_PATH)
    return detector


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model_loaded": detector is not None}


@app.get("/classes")
def get_classes() -> dict:
    return {"classes": DEFECT_LABELS}


@app.post("/detect")
async def detect(file: Annotated[UploadFile, File(...)]) -> dict:
    """Detect weld defects in an uploaded image.

    Returns bounding boxes, class labels, and confidence scores
    for each detected defect.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    det = get_detector()
    detections = det.detect(image)

    return {
        "filename": file.filename,
        "image_size": {"width": image.width, "height": image.height},
        "num_detections": len(detections),
        "detections": detections,
    }


@app.post("/detect/visualize")
async def detect_and_visualize(file: Annotated[UploadFile, File(...)]) -> Response:
    """Detect defects and return annotated image with bounding boxes drawn."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    det = get_detector()
    detections = det.detect(image)

    import cv2

    annotated = draw_detections(image, detections)
    _, buffer = cv2.imencode(".png", annotated)

    return Response(content=buffer.tobytes(), media_type="image/png")
