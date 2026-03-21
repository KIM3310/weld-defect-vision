"""CNN-based welding defect classifier using ResNet18 backbone with transfer learning."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defect taxonomy
# ---------------------------------------------------------------------------


class DefectType(str, Enum):
    """ISO 6520-1 aligned welding defect categories."""

    CRACK = "crack"
    POROSITY = "porosity"
    UNDERCUT = "undercut"
    INCOMPLETE_FUSION = "incomplete_fusion"
    OVERLAP = "overlap"
    SPATTER = "spatter"
    NO_DEFECT = "no_defect"


DEFECT_DESCRIPTIONS: dict[DefectType, str] = {
    DefectType.CRACK: "Linear discontinuity caused by thermal stress or hydrogen embrittlement",
    DefectType.POROSITY: "Spherical gas pockets trapped during solidification",
    DefectType.UNDERCUT: "Groove melted into base metal adjacent to weld toe",
    DefectType.INCOMPLETE_FUSION: "Lack of fusion between weld metal and base material",
    DefectType.OVERLAP: "Protrusion of weld metal beyond the weld toe",
    DefectType.SPATTER: "Metal particles expelled during welding process",
    DefectType.NO_DEFECT: "Weld meets quality acceptance criteria",
}

NUM_CLASSES = len(DefectType)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """Output of a single defect detection inference."""

    defect_type: DefectType
    confidence: float
    class_probabilities: dict[str, float] = field(default_factory=dict)
    description: str = ""
    demo_mode: bool = False

    def __post_init__(self) -> None:
        if not self.description:
            self.description = DEFECT_DESCRIPTIONS.get(self.defect_type, "")

    @property
    def is_defect(self) -> bool:
        return self.defect_type != DefectType.NO_DEFECT


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class WeldDefectCNN(nn.Module):
    """ResNet18 backbone with a fine-tuned classification head for weld defect detection."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Freeze early convolutional layers; fine-tune from layer3 onward
        for name, param in backbone.named_parameters():
            if not any(name.startswith(layer) for layer in ("layer3", "layer4", "fc")):
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Preprocessing transform
# ---------------------------------------------------------------------------

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------


class DefectClassifier:
    """High-level interface for defect classification.

    Operates in two modes:
    - **Model mode**: uses a loaded PyTorch checkpoint for inference.
    - **Demo mode**: applies rule-based heuristics on image statistics
      (useful for CI/CD, demos, and environments without a trained model).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        demo_mode: bool = False,
    ) -> None:
        self.demo_mode = demo_mode
        self.device = torch.device(
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model: WeldDefectCNN | None = None
        self._classes = list(DefectType)

        if not demo_mode:
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str | Path | None) -> None:
        """Attempt to load a trained checkpoint; fall back to demo mode."""
        self._model = WeldDefectCNN(num_classes=NUM_CLASSES, pretrained=False)

        if model_path is not None:
            path = Path(model_path)
            if path.exists():
                try:
                    state = torch.load(path, map_location=self.device, weights_only=True)
                    self._model.load_state_dict(state)
                    logger.info("Loaded model weights from %s", path)
                except Exception as exc:
                    logger.warning("Could not load weights from %s: %s – using demo mode", path, exc)
                    self.demo_mode = True
            else:
                logger.info("No checkpoint at %s – activating demo mode", path)
                self.demo_mode = True
        else:
            logger.info("No model_path provided – activating demo mode")
            self.demo_mode = True

        if not self.demo_mode and self._model is not None:
            self._model.to(self.device)
            self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image) -> DetectionResult:
        """Run defect classification on a PIL image."""
        if self.demo_mode:
            return self._demo_predict(image)
        return self._model_predict(image)

    def predict_batch(self, images: list[Image.Image]) -> list[DetectionResult]:
        """Run classification on a list of PIL images."""
        return [self.predict(img) for img in images]

    # ------------------------------------------------------------------
    # Inference paths
    # ------------------------------------------------------------------

    def _model_predict(self, image: Image.Image) -> DetectionResult:
        assert self._model is not None
        tensor = INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        defect_type = self._classes[pred_idx]
        confidence = float(probs[pred_idx])
        class_probs = {cls.value: float(p) for cls, p in zip(self._classes, probs)}

        return DetectionResult(
            defect_type=defect_type,
            confidence=confidence,
            class_probabilities=class_probs,
            demo_mode=False,
        )

    def _demo_predict(self, image: Image.Image) -> DetectionResult:
        """Rule-based heuristics on image statistics for demo/testing purposes."""
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        mean_intensity = float(gray.mean())
        std_intensity = float(gray.std())
        dark_pixel_ratio = float((gray < 50).mean())
        bright_pixel_ratio = float((gray > 200).mean())

        probs = self._heuristic_probs(mean_intensity, std_intensity, dark_pixel_ratio, bright_pixel_ratio, image)
        pred_idx = int(np.argmax(probs))
        defect_type = self._classes[pred_idx]

        class_probs = {cls.value: float(p) for cls, p in zip(self._classes, probs)}
        return DetectionResult(
            defect_type=defect_type,
            confidence=float(probs[pred_idx]),
            class_probabilities=class_probs,
            demo_mode=True,
        )

    def _heuristic_probs(
        self,
        mean: float,
        std: float,
        dark_ratio: float,
        bright_ratio: float,
        image: Image.Image | None = None,
    ) -> np.ndarray:
        """Map image statistics to per-class probability estimates."""
        scores = np.zeros(NUM_CLASSES, dtype=np.float32)
        idx = {d: i for i, d in enumerate(DefectType)}

        # Crack: detect thin dark lines via vertical connectivity analysis
        crack_score = min(1.0, std / 40.0) * min(1.0, dark_ratio * 80)
        if image is not None:
            gray_arr = np.array(image.convert("L"), dtype=np.float32)
            dark_mask = gray_arr < 50
            # Check vertical connectivity: count max consecutive dark pixels per column
            max_run = 0
            for col in range(dark_mask.shape[1]):
                run = 0
                for row in range(dark_mask.shape[0]):
                    if dark_mask[row, col]:
                        run += 1
                        max_run = max(max_run, run)
                    else:
                        run = 0
            line_score = min(1.0, max_run / 30.0)
            crack_score = max(crack_score, line_score)
        scores[idx[DefectType.CRACK]] = crack_score

        # Porosity: clustered dark spots → dark_ratio moderate
        scores[idx[DefectType.POROSITY]] = min(1.0, dark_ratio * 10) * (1 - min(1.0, dark_ratio * 40))

        # Undercut: dark edge regions, lower mean
        scores[idx[DefectType.UNDERCUT]] = max(0.0, 1.0 - mean / 150.0) * min(1.0, dark_ratio * 15)

        # Incomplete fusion: low mean intensity overall
        scores[idx[DefectType.INCOMPLETE_FUSION]] = max(0.0, 1.0 - mean / 100.0)

        # Overlap: bright protrusions
        scores[idx[DefectType.OVERLAP]] = min(1.0, bright_ratio * 8)

        # Spatter: many small bright spots
        scores[idx[DefectType.SPATTER]] = min(1.0, bright_ratio * 12) * min(1.0, std / 60.0)

        # No defect: uniform, moderate intensity
        uniformity = max(0.0, 1.0 - std / 50.0)
        moderation = 1.0 - abs(mean - 128) / 128.0
        scores[idx[DefectType.NO_DEFECT]] = uniformity * moderation

        # Softmax normalisation
        scores = np.exp(scores * 3.0)
        return scores / scores.sum()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        return [d.value for d in self._classes]

    def get_model_info(self) -> dict[str, Any]:
        return {
            "mode": "demo" if self.demo_mode else "model",
            "device": str(self.device),
            "num_classes": NUM_CLASSES,
            "classes": self.class_names,
        }
