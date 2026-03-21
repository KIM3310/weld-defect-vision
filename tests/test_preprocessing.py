"""Tests for the image preprocessing pipeline."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.preprocessing.pipeline import PreprocessingPipeline, PreprocessingResult
from tests.conftest import make_image_no_defect, make_image_porosity


class TestPreprocessingPipelineInit:
    def test_default_config(self) -> None:
        p = PreprocessingPipeline()
        assert p.target_size == (224, 224)
        assert p.apply_clahe is True
        assert p.apply_noise_reduction is False

    def test_custom_config(self) -> None:
        p = PreprocessingPipeline(
            target_size=(128, 128),
            apply_clahe=False,
            apply_noise_reduction=True,
            noise_kernel_size=5,
        )
        assert p.target_size == (128, 128)
        assert p.apply_clahe is False
        assert p.apply_noise_reduction is True
        assert p.noise_kernel_size == 5

    def test_get_config_returns_dict(self) -> None:
        p = PreprocessingPipeline()
        cfg = p.get_config()
        assert isinstance(cfg, dict)
        assert "target_size" in cfg
        assert "apply_clahe" in cfg


class TestPreprocessingPipelineProcess:
    def test_process_pil_image(self, pipeline: PreprocessingPipeline) -> None:
        img = make_image_no_defect()
        result = pipeline.process(img)
        assert isinstance(result, PreprocessingResult)

    def test_output_is_pil_image(self, pipeline: PreprocessingPipeline) -> None:
        img = make_image_no_defect()
        result = pipeline.process(img)
        assert isinstance(result.image, Image.Image)

    def test_output_size_matches_target(self, pipeline: PreprocessingPipeline) -> None:
        img = make_image_no_defect(h=512, w=512)
        result = pipeline.process(img)
        assert result.processed_size == (224, 224)

    def test_process_bytes(self, pipeline: PreprocessingPipeline) -> None:
        buf = io.BytesIO()
        make_image_no_defect().save(buf, format="PNG")
        raw = buf.getvalue()
        result = pipeline.process_bytes(raw)
        assert isinstance(result, PreprocessingResult)
        assert result.processed_size == (224, 224)

    def test_process_numpy_array(self, pipeline: PreprocessingPipeline) -> None:
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 128
        result = pipeline.process(arr)
        assert isinstance(result, PreprocessingResult)

    def test_process_grayscale_numpy(self, pipeline: PreprocessingPipeline) -> None:
        arr = np.ones((100, 100), dtype=np.uint8) * 128
        result = pipeline.process(arr)
        assert isinstance(result.image, Image.Image)

    def test_original_size_recorded(self, pipeline: PreprocessingPipeline) -> None:
        img = make_image_no_defect(h=300, w=400)
        result = pipeline.process(img)
        assert result.original_size == (400, 300)  # PIL uses (W, H)

    def test_clahe_flag_set(self) -> None:
        p = PreprocessingPipeline(apply_clahe=True)
        result = p.process(make_image_no_defect())
        assert result.clahe_applied is True

    def test_clahe_flag_not_set_when_disabled(self) -> None:
        p = PreprocessingPipeline(apply_clahe=False)
        result = p.process(make_image_no_defect())
        assert result.clahe_applied is False

    def test_noise_reduction_flag(self) -> None:
        p = PreprocessingPipeline(apply_noise_reduction=True, noise_kernel_size=3)
        result = p.process(make_image_no_defect())
        assert result.noise_reduction_applied is True

    def test_steps_applied_not_empty(self, pipeline: PreprocessingPipeline) -> None:
        result = pipeline.process(make_image_no_defect())
        assert len(result.steps_applied) > 0

    def test_output_mode_is_rgb(self, pipeline: PreprocessingPipeline) -> None:
        result = pipeline.process(make_image_no_defect())
        assert result.image.mode == "RGB"

    def test_to_dict_structure(self, pipeline: PreprocessingPipeline) -> None:
        result = pipeline.process(make_image_no_defect())
        d = result.to_dict()
        assert "original_size" in d
        assert "processed_size" in d
        assert "clahe_applied" in d
        assert "steps_applied" in d

    def test_unsupported_type_raises(self, pipeline: PreprocessingPipeline) -> None:
        with pytest.raises(TypeError):
            pipeline.process("not_an_image")  # type: ignore[arg-type]

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
    def test_process_various_formats(self, pipeline: PreprocessingPipeline, fmt: str) -> None:
        buf = io.BytesIO()
        img = make_image_no_defect()
        img.save(buf, format=fmt)
        result = pipeline.process_bytes(buf.getvalue())
        assert result.processed_size == (224, 224)

    def test_even_kernel_size_corrected(self) -> None:
        """Even noise_kernel_size should be auto-corrected to odd."""
        p = PreprocessingPipeline(apply_noise_reduction=True, noise_kernel_size=4)
        result = p.process(make_image_porosity())
        assert result.noise_reduction_applied is True

    def test_pipeline_does_not_mutate_input(self, pipeline: PreprocessingPipeline) -> None:
        img = make_image_no_defect()
        original_size = img.size
        pipeline.process(img)
        assert img.size == original_size
