import numpy as np
import pytest
import torch
from PIL import Image
from ultralytics.engine.results import Results

from src.inference import WeldDefectDetector


class StaticResultModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, **_kwargs):
        return self.predictions


def build_result(boxes: torch.Tensor | None = None) -> Results:
    return Results(
        orig_img=np.zeros((16, 16, 3), dtype=np.uint8),
        path="fixture.png",
        names={0: "crack"},
        boxes=boxes,
    )


def build_detector(predictions) -> WeldDefectDetector:
    detector = WeldDefectDetector.__new__(WeldDefectDetector)
    detector.model = StaticResultModel(predictions)
    detector.conf_threshold = 0.25
    detector.iou_threshold = 0.45
    detector.device = None
    return detector


def test_detect_handles_result_without_boxes():
    detector = build_detector([build_result()])

    detections = detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))

    assert detections == []


def test_detect_batch_preserves_empty_result():
    detector = build_detector([build_result()])

    detections = detector.detect_batch([Image.new("RGB", (16, 16))])

    assert detections == [[]]


def test_detect_parses_ultralytics_detection_result():
    boxes = torch.tensor([[1.0, 2.0, 9.0, 10.0, 0.875, 0.0]])
    detector = build_detector([build_result(boxes)])

    detections = detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))

    assert detections == [
        {
            "bbox": [1.0, 2.0, 9.0, 10.0],
            "class_id": 0,
            "class_name": "Crack",
            "confidence": 0.875,
        }
    ]


def test_detect_accepts_iterator_results():
    detector = build_detector(iter([build_result()]))

    detections = detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))

    assert detections == []


def test_detect_rejects_tensor_embeddings():
    detector = build_detector([torch.tensor([[0.1, 0.2]])])

    with pytest.raises(TypeError, match="Tensor embedding"):
        detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))


def test_detect_rejects_streamed_tensor_embeddings():
    detector = build_detector(iter([torch.tensor([[0.1, 0.2]])]))

    with pytest.raises(TypeError, match="Tensor embedding"):
        detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))
