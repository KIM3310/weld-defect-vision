from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.inference import WeldDefectDetector


class EmptyResultModel:
    def predict(self, **_kwargs):
        return [SimpleNamespace(boxes=None)]


def build_empty_result_detector() -> WeldDefectDetector:
    detector = WeldDefectDetector.__new__(WeldDefectDetector)
    detector.model = EmptyResultModel()
    detector.conf_threshold = 0.25
    detector.iou_threshold = 0.45
    detector.device = None
    return detector


def test_detect_handles_result_without_boxes():
    detector = build_empty_result_detector()

    detections = detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))

    assert detections == []


def test_detect_batch_preserves_empty_result():
    detector = build_empty_result_detector()

    detections = detector.detect_batch([Image.new("RGB", (16, 16))])

    assert detections == [[]]
