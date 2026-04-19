"""Single-image inference pipeline for weld defect detection."""

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.config import DEFECT_LABELS


class WeldDefectDetector:
    """Inference wrapper for weld defect detection.

    Loads a trained YOLOv8 model and provides methods for
    single-image and batch detection with NMS filtering.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ):
        self.model = YOLO(str(checkpoint_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def detect(self, image: Image.Image | np.ndarray) -> list[dict]:
        """Detect weld defects in a single image.

        Returns list of detections, each with:
        - bbox: [x1, y1, x2, y2] in pixels
        - class_id: integer class index
        - class_name: human-readable defect type
        - confidence: detection confidence score
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                cls_id = int(boxes.cls[i].cpu().item())
                conf = float(boxes.conf[i].cpu().item())

                detections.append(
                    {
                        "bbox": [round(c, 1) for c in bbox],
                        "class_id": cls_id,
                        "class_name": DEFECT_LABELS.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(conf, 4),
                    }
                )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def detect_from_path(self, image_path: str | Path) -> dict:
        """Detect defects from a file path."""
        image = Image.open(image_path).convert("RGB")
        detections = self.detect(image)
        return {
            "image_path": str(image_path),
            "num_detections": len(detections),
            "detections": detections,
        }

    def detect_batch(self, images: list[Image.Image]) -> list[list[dict]]:
        """Detect defects in a batch of images."""
        results = self.model.predict(
            source=images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        batch_detections = []
        for result in results:
            detections = []
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                cls_id = int(boxes.cls[i].cpu().item())
                conf = float(boxes.conf[i].cpu().item())
                detections.append(
                    {
                        "bbox": [round(c, 1) for c in bbox],
                        "class_id": cls_id,
                        "class_name": DEFECT_LABELS.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(conf, 4),
                    }
                )
            batch_detections.append(detections)

        return batch_detections
