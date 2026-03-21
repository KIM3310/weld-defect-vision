"""Tests for the DefectClassifier model."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.models.classifier import (
    DefectClassifier,
    DefectType,
    DetectionResult,
    WeldDefectCNN,
)
from tests.conftest import (
    make_image_crack,
    make_image_no_defect,
    make_image_porosity,
    make_image_spatter,
    make_image_undercut,
)


class TestDefectType:
    def test_all_expected_types_exist(self) -> None:
        expected = {"crack", "porosity", "undercut", "incomplete_fusion", "overlap", "spatter", "no_defect"}
        actual = {d.value for d in DefectType}
        assert actual == expected

    def test_string_enum_values(self) -> None:
        assert DefectType.CRACK == "crack"
        assert DefectType.NO_DEFECT == "no_defect"


class TestDetectionResult:
    def test_is_defect_true_for_defects(self) -> None:
        for defect in DefectType:
            result = DetectionResult(defect_type=defect, confidence=0.9)
            if defect == DefectType.NO_DEFECT:
                assert not result.is_defect
            else:
                assert result.is_defect

    def test_description_auto_populated(self) -> None:
        result = DetectionResult(defect_type=DefectType.CRACK, confidence=0.8)
        assert len(result.description) > 0
        assert "crack" in result.description.lower() or "linear" in result.description.lower()

    def test_custom_description_preserved(self) -> None:
        result = DetectionResult(
            defect_type=DefectType.CRACK,
            confidence=0.8,
            description="custom desc",
        )
        assert result.description == "custom desc"

    def test_class_probabilities_defaults_empty(self) -> None:
        result = DetectionResult(defect_type=DefectType.POROSITY, confidence=0.7)
        assert result.class_probabilities == {}


class TestDefectClassifierInit:
    def test_demo_mode_explicit(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        assert clf.demo_mode is True

    def test_no_path_falls_back_to_demo(self) -> None:
        clf = DefectClassifier(model_path=None)
        assert clf.demo_mode is True

    def test_nonexistent_path_falls_back_to_demo(self) -> None:
        clf = DefectClassifier(model_path="/nonexistent/model.pt")
        assert clf.demo_mode is True

    def test_class_names_correct(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        assert set(clf.class_names) == {d.value for d in DefectType}

    def test_get_model_info_structure(self) -> None:
        clf = DefectClassifier(demo_mode=True)
        info = clf.get_model_info()
        assert info["mode"] == "demo"
        assert info["num_classes"] == len(DefectType)
        assert isinstance(info["classes"], list)


class TestDefectClassifierPredict:
    def test_predict_returns_detection_result(self, classifier: DefectClassifier) -> None:
        img = make_image_no_defect()
        result = classifier.predict(img)
        assert isinstance(result, DetectionResult)

    def test_predict_confidence_in_range(self, classifier: DefectClassifier) -> None:
        for factory in (make_image_no_defect, make_image_porosity, make_image_crack):
            img = factory()
            result = classifier.predict(img)
            assert 0.0 <= result.confidence <= 1.0

    def test_predict_class_probs_sum_to_one(self, classifier: DefectClassifier) -> None:
        img = make_image_porosity()
        result = classifier.predict(img)
        total = sum(result.class_probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_predict_all_classes_in_probs(self, classifier: DefectClassifier) -> None:
        img = make_image_no_defect()
        result = classifier.predict(img)
        assert set(result.class_probabilities.keys()) == {d.value for d in DefectType}

    def test_predict_demo_mode_flag_set(self, classifier: DefectClassifier) -> None:
        img = make_image_no_defect()
        result = classifier.predict(img)
        assert result.demo_mode is True

    def test_predict_valid_defect_type(self, classifier: DefectClassifier) -> None:
        img = make_image_crack()
        result = classifier.predict(img)
        assert result.defect_type in DefectType

    def test_predict_batch_length(self, classifier: DefectClassifier) -> None:
        images = [make_image_no_defect(), make_image_porosity(), make_image_crack()]
        results = classifier.predict_batch(images)
        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_predict_rgb_conversion(self, classifier: DefectClassifier) -> None:
        """Grayscale input should be handled correctly."""
        gray = Image.fromarray(np.full((224, 224), 128, dtype=np.uint8), mode="L")
        result = classifier.predict(gray)
        assert isinstance(result, DetectionResult)

    def test_no_defect_image_lower_score(self, classifier: DefectClassifier) -> None:
        """Uniform image should rank no_defect relatively high."""
        no_defect = make_image_no_defect()
        result = classifier.predict(no_defect)
        nd_prob = result.class_probabilities.get("no_defect", 0.0)
        # In demo mode no_defect should be among the top-2 candidates for a uniform image
        sorted_probs = sorted(result.class_probabilities.values(), reverse=True)
        assert nd_prob >= sorted_probs[1] - 0.05  # within top-2 with tolerance

    @pytest.mark.parametrize("size", [(32, 32), (100, 100), (512, 512)])
    def test_predict_various_input_sizes(
        self, classifier: DefectClassifier, size: tuple[int, int]
    ) -> None:
        arr = np.ones((*size, 3), dtype=np.uint8) * 150
        img = Image.fromarray(arr)
        result = classifier.predict(img)
        assert isinstance(result, DetectionResult)


class TestWeldDefectCNN:
    def test_model_instantiation(self) -> None:
        import torch

        model = WeldDefectCNN(num_classes=7, pretrained=False)
        assert model is not None

    def test_forward_pass_shape(self) -> None:
        import torch

        model = WeldDefectCNN(num_classes=7, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 7)

    def test_frozen_early_layers(self) -> None:
        model = WeldDefectCNN(num_classes=7, pretrained=False)
        # layer1 and layer2 should be frozen
        for name, param in model.named_parameters():
            if "layer1" in name or "layer2" in name:
                assert not param.requires_grad, f"Expected {name} to be frozen"
