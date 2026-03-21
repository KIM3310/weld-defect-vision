"""Image preprocessing pipeline for weld inspection images.

Steps:
1. Decode and validate input (PIL / bytes / numpy array)
2. Resize to model input resolution
3. Histogram equalization (CLAHE) for contrast enhancement
4. Optional noise reduction (Gaussian blur)
5. Weld region extraction via thresholding (optional)
6. Normalisation metadata for traceability
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingResult:
    """Container for a preprocessed image and its metadata."""

    image: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    clahe_applied: bool
    noise_reduction_applied: bool
    steps_applied: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_size": list(self.original_size),
            "processed_size": list(self.processed_size),
            "clahe_applied": self.clahe_applied,
            "noise_reduction_applied": self.noise_reduction_applied,
            "steps_applied": self.steps_applied,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """Configurable preprocessing pipeline for weld inspection images.

    Args:
        target_size: (width, height) to resize to. Default (224, 224).
        apply_clahe: Apply CLAHE contrast enhancement. Default True.
        clahe_clip_limit: Clip limit for CLAHE. Default 2.0.
        clahe_tile_grid: Tile grid size for CLAHE. Default (8, 8).
        apply_noise_reduction: Apply Gaussian denoising. Default False.
        noise_kernel_size: Kernel size for Gaussian blur. Default 3.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
        apply_noise_reduction: bool = False,
        noise_kernel_size: int = 3,
    ) -> None:
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.apply_noise_reduction = apply_noise_reduction
        self.noise_kernel_size = noise_kernel_size

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def process(self, source: Image.Image | bytes | np.ndarray) -> PreprocessingResult:
        """Preprocess an image from any supported source type."""
        image = self._to_pil(source)
        return self._run_pipeline(image)

    def process_bytes(self, data: bytes) -> PreprocessingResult:
        """Preprocess raw image bytes (e.g. from an HTTP upload)."""
        return self.process(data)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_pipeline(self, image: Image.Image) -> PreprocessingResult:
        original_size = image.size  # (W, H)
        steps: list[str] = []

        # 1. Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            steps.append(f"convert_to_rgb (was {image.mode})")

        # 2. Auto-orient based on EXIF
        image = ImageOps.exif_transpose(image)
        steps.append("exif_transpose")

        # 3. Resize
        if image.size != self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            steps.append(f"resize_{self.target_size[0]}x{self.target_size[1]}")

        # 4. CLAHE contrast enhancement (applied per channel in LAB space)
        clahe_applied = False
        if self.apply_clahe:
            image = self._apply_clahe(image)
            clahe_applied = True
            steps.append("clahe_lab")

        # 5. Gaussian noise reduction
        noise_applied = False
        if self.apply_noise_reduction:
            image = self._apply_noise_reduction(image)
            noise_applied = True
            steps.append(f"gaussian_blur_k{self.noise_kernel_size}")

        return PreprocessingResult(
            image=image,
            original_size=original_size,
            processed_size=image.size,
            clahe_applied=clahe_applied,
            noise_reduction_applied=noise_applied,
            steps_applied=steps,
        )

    def _apply_clahe(self, image: Image.Image) -> Image.Image:
        """Apply CLAHE in LAB colour space to enhance weld feature contrast."""
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_eq = self._clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        rgb = cv2.cvtColor(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _apply_noise_reduction(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur for noise reduction."""
        k = self.noise_kernel_size
        if k % 2 == 0:
            k += 1  # kernel must be odd
        arr = cv2.GaussianBlur(np.array(image), (k, k), 0)
        return Image.fromarray(arr)

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(source: Image.Image | bytes | np.ndarray) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.copy()
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source))
        if isinstance(source, np.ndarray):
            if source.dtype != np.uint8:
                source = (source * 255).clip(0, 255).astype(np.uint8)
            if source.ndim == 2:
                return Image.fromarray(source, mode="L")
            return Image.fromarray(source)
        raise TypeError(f"Unsupported source type: {type(source)}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        return {
            "target_size": list(self.target_size),
            "apply_clahe": self.apply_clahe,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid": list(self.clahe_tile_grid),
            "apply_noise_reduction": self.apply_noise_reduction,
            "noise_kernel_size": self.noise_kernel_size,
        }
