"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel


class BoundingBox(BaseModel):
    bbox: list[float]
    class_id: int
    class_name: str
    confidence: float


class DetectionResponse(BaseModel):
    filename: str | None = None
    image_size: dict[str, int]
    num_detections: int
    detections: list[BoundingBox]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ClassesResponse(BaseModel):
    classes: dict[int, str]
