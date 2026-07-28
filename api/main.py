"""FastAPI inference server for weld defect detection."""

import asyncio
import io
import os
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, TypeVar

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from starlette.types import ASGIApp, Receive, Scope, Send

from src.config import DEFECT_LABELS
from src.inference import Detection, WeldDefectDetector
from src.visualize import draw_detections

T = TypeVar("T")


class UploadBodyLimitMiddleware:
    """Reject oversized image uploads before multipart parsing when possible."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in {"/detect", "/detect/visualize"}
        ):
            await self.app(scope, receive, send)
            return

        max_body_bytes = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_body_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                pass

        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > max_body_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Request body exceeds the image upload limit",
                    )
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": "Request body exceeds the image upload limit"},
        )
        await response(scope, receive, send)


app = FastAPI(
    title="Weld Defect Vision",
    description="Weld defect detection API using YOLOv8",
    version="0.1.0",
)

CHECKPOINT_PATH = Path("checkpoints/best.pt")
MAX_UPLOAD_BYTES = int(os.getenv("WELD_MAX_UPLOAD_BYTES", str(16 * 1024 * 1024)))
MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_IMAGE_PIXELS = int(os.getenv("WELD_MAX_IMAGE_PIXELS", "50000000"))
MAX_CONCURRENT_INFERENCES = int(os.getenv("WELD_MAX_CONCURRENT_INFERENCES", "1"))

if min(MAX_UPLOAD_BYTES, MAX_IMAGE_PIXELS, MAX_CONCURRENT_INFERENCES) < 1:
    raise RuntimeError("Weld API resource limits must be positive integers")

detector: WeldDefectDetector | None = None
_detector_lock = threading.Lock()
_inference_slots = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCES)
INVALID_IMAGE_DETAIL = "Uploaded file is not a valid or supported raster image"
app.add_middleware(UploadBodyLimitMiddleware)


def get_detector() -> WeldDefectDetector:
    global detector
    if detector is not None:
        return detector

    with _detector_lock:
        if detector is None:
            if not CHECKPOINT_PATH.exists():
                raise HTTPException(
                    status_code=503,
                    detail="Model checkpoint not found. Train the model first.",
                )
            detector = WeldDefectDetector(CHECKPOINT_PATH)
    return detector


async def _read_upload(file: UploadFile) -> bytes:
    try:
        if file.size is not None and file.size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {MAX_UPLOAD_BYTES}-byte upload limit",
            )
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES}-byte upload limit",
        )
    if not contents:
        raise HTTPException(status_code=422, detail=INVALID_IMAGE_DETAIL)
    return contents


def _validate_image(contents: bytes) -> None:
    try:
        with Image.open(io.BytesIO(contents)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit",
                )
            source.verify()
    except HTTPException:
        raise
    except Image.DecompressionBombError as exc:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit",
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=INVALID_IMAGE_DETAIL) from exc


def _decode_image(contents: bytes) -> Image.Image:
    _validate_image(contents)
    try:
        with Image.open(io.BytesIO(contents)) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=INVALID_IMAGE_DETAIL) from exc


async def decode_image_upload(file: UploadFile) -> Image.Image:
    """Validate and fully decode an uploaded raster image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        await file.close()
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await _read_upload(file)
    return await asyncio.to_thread(_decode_image, contents)


def _run_with_inference_slot(operation: Callable[[], T]) -> T:
    with _inference_slots:
        return operation()


async def _run_inference(operation: Callable[[], T]) -> T:
    return await asyncio.to_thread(_run_with_inference_slot, operation)


def _detect_image(image: Image.Image) -> list[Detection]:
    return get_detector().detect(image)


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
    image = await decode_image_upload(file)

    try:
        image_size = {"width": image.width, "height": image.height}
        detections = await _run_inference(partial(_detect_image, image))
    finally:
        image.close()

    return {
        "filename": file.filename,
        "image_size": image_size,
        "num_detections": len(detections),
        "detections": detections,
    }


@app.post("/detect/visualize")
async def detect_and_visualize(file: Annotated[UploadFile, File(...)]) -> Response:
    """Detect defects and return annotated image with bounding boxes drawn."""
    image = await decode_image_upload(file)

    try:
        detections = await _run_inference(partial(_detect_image, image))

        import cv2

        annotated = draw_detections(image, detections)
        _, buffer = cv2.imencode(".png", annotated)
    finally:
        image.close()

    return Response(content=buffer.tobytes(), media_type="image/png")
