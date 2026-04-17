"""Configuration for weld defect detection."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    data_yaml: Path = Path("data/weld_defect.yaml")
    output_dir: Path = Path("runs/detect")
    checkpoint_dir: Path = Path("checkpoints")

    # Model
    base_model: str = "yolov8n.pt"  # nano for fast training, swap to yolov8s/m for production
    num_classes: int = 5

    # Training
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.01
    patience: int = 20
    workers: int = 4
    seed: int = 42

    # Augmentation (built into YOLO)
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.5
    fliplr: float = 0.5

    # NMS
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100

    # Classes
    class_names: list[str] = field(
        default_factory=lambda: [
            "crack",
            "porosity",
            "spatter",
            "undercut",
            "overlap",
        ]
    )


DEFECT_LABELS = {
    0: "Crack",
    1: "Porosity",
    2: "Spatter",
    3: "Undercut",
    4: "Overlap",
}

DEFECT_COLORS = {
    0: (255, 0, 0),      # Red
    1: (0, 255, 0),      # Green
    2: (0, 0, 255),      # Blue
    3: (255, 165, 0),    # Orange
    4: (128, 0, 128),    # Purple
}
