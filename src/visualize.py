"""Visualization utilities for detection results."""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import DEFECT_COLORS, DEFECT_LABELS


def draw_detections(
    image: np.ndarray | Image.Image,
    detections: list[dict],
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: Input image (numpy BGR or PIL RGB)
        detections: List of detection dicts from WeldDefectDetector
        line_thickness: Bounding box line thickness

    Returns:
        Annotated image as numpy array (BGR)
    """
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        cls_id = det["class_id"]
        conf = det["confidence"]
        label = f"{det['class_name']} {conf:.2f}"

        color = DEFECT_COLORS.get(cls_id, (255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def save_detection_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_path: str | Path,
    cols: int = 3,
) -> None:
    """Save a grid of detection result images."""
    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n:
                rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
                axes[i][j].imshow(rgb)
                axes[i][j].set_title(titles[idx], fontsize=10)
            axes[i][j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Detection grid saved: {output_path}")


def create_detection_summary(detections: list[dict]) -> dict:
    """Summarize detections by defect type."""
    summary: dict[str, list[float]] = {}
    for det in detections:
        name = det["class_name"]
        if name not in summary:
            summary[name] = []
        summary[name].append(det["confidence"])

    return {
        name: {
            "count": len(confs),
            "avg_confidence": round(sum(confs) / len(confs), 4),
            "max_confidence": round(max(confs), 4),
        }
        for name, confs in summary.items()
    }
