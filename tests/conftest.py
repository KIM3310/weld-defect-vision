"""Shared pytest fixtures and synthetic image generators."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.models.classifier import DefectClassifier, DefectType
from app.models.severity import SeverityScorer
from app.preprocessing.pipeline import PreprocessingPipeline
from app.reporting.generator import ReportGenerator

# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(2024)


def _base_weld_array(h: int = 224, w: int = 224) -> np.ndarray:
    """Gray steel plate with a bright weld bead in the middle third."""
    arr = np.ones((h, w, 3), dtype=np.uint8) * 160
    bead_start = h // 3
    bead_end = (2 * h) // 3
    arr[bead_start:bead_end, :] = 200
    return arr


def make_image_no_defect(h: int = 224, w: int = 224) -> Image.Image:
    arr = _base_weld_array(h, w)
    return Image.fromarray(arr)


def make_image_porosity(h: int = 224, w: int = 224, n_pores: int = 12) -> Image.Image:
    arr = _base_weld_array(h, w)
    bead_start, bead_end = h // 3, (2 * h) // 3
    for _ in range(n_pores):
        cx = int(_RNG.integers(10, w - 10))
        cy = int(_RNG.integers(bead_start + 5, bead_end - 5))
        r = int(_RNG.integers(4, 10))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    py, px = cy + dy, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        arr[py, px] = 25
    return Image.fromarray(arr)


def make_image_crack(h: int = 224, w: int = 224) -> Image.Image:
    arr = _base_weld_array(h, w)
    bead_start = h // 3
    x = w // 2
    for dy in range(int(h / 3)):
        jitter = int(_RNG.integers(-1, 2))
        x = min(w - 2, max(1, x + jitter))
        arr[bead_start + dy, x - 3 : x + 4] = 5
    # Add secondary crack branch for stronger signal
    x2 = w // 2 + 15
    for dy in range(int(h / 5)):
        jitter = int(_RNG.integers(-1, 2))
        x2 = min(w - 2, max(1, x2 + jitter))
        arr[bead_start + dy, x2 - 2 : x2 + 3] = 5
    return Image.fromarray(arr)


def make_image_undercut(h: int = 224, w: int = 224) -> Image.Image:
    arr = _base_weld_array(h, w)
    bead_start, bead_end = h // 3, (2 * h) // 3
    arr[bead_start : bead_start + 3, :] = 40
    arr[bead_end - 3 : bead_end, :] = 40
    return Image.fromarray(arr)


def make_image_spatter(h: int = 224, w: int = 224, n_spots: int = 30) -> Image.Image:
    arr = _base_weld_array(h, w)
    for _ in range(n_spots):
        sx = int(_RNG.integers(5, w - 5))
        sy = int(_RNG.integers(5, h - 5))
        r = int(_RNG.integers(2, 5))
        arr[max(0, sy - r) : sy + r, max(0, sx - r) : sx + r] = 245
    return Image.fromarray(arr)


def make_image_incomplete_fusion(h: int = 224, w: int = 224) -> Image.Image:
    arr = _base_weld_array(h, w)
    arr[:, :] = 40  # very dark → low mean
    arr[h // 3 : (2 * h) // 3, :] = 60
    return Image.fromarray(arr)


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


_DEFECT_FACTORIES = {
    DefectType.NO_DEFECT: make_image_no_defect,
    DefectType.POROSITY: make_image_porosity,
    DefectType.CRACK: make_image_crack,
    DefectType.UNDERCUT: make_image_undercut,
    DefectType.SPATTER: make_image_spatter,
    DefectType.INCOMPLETE_FUSION: make_image_incomplete_fusion,
}


def generate_all_samples(output_dir: str | Path = "data/sample") -> None:
    """Write one synthetic PNG per defect type into output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for defect, factory in _DEFECT_FACTORIES.items():
        img = factory()
        img.save(out / f"{defect.value}.png")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def classifier() -> DefectClassifier:
    return DefectClassifier(demo_mode=True)


@pytest.fixture(scope="session")
def scorer() -> SeverityScorer:
    return SeverityScorer()


@pytest.fixture(scope="session")
def pipeline() -> PreprocessingPipeline:
    return PreprocessingPipeline(target_size=(224, 224), apply_clahe=True)


@pytest.fixture(scope="session")
def reporter() -> ReportGenerator:
    return ReportGenerator()


@pytest.fixture
def sample_no_defect() -> Image.Image:
    return make_image_no_defect()


@pytest.fixture
def sample_porosity() -> Image.Image:
    return make_image_porosity()


@pytest.fixture
def sample_crack() -> Image.Image:
    return make_image_crack()


@pytest.fixture
def sample_undercut() -> Image.Image:
    return make_image_undercut()


@pytest.fixture
def sample_spatter() -> Image.Image:
    return make_image_spatter()


@pytest.fixture
def no_defect_bytes() -> bytes:
    return image_to_bytes(make_image_no_defect())


@pytest.fixture
def porosity_bytes() -> bytes:
    return image_to_bytes(make_image_porosity())


@pytest.fixture
def crack_bytes() -> bytes:
    return image_to_bytes(make_image_crack())
