"""Minimal Triton client for the weld_defect model.

Demonstrates:
    - Connecting to Triton over gRPC
    - Preprocessing an image into the model input tensor
    - Running inference
    - Post-processing YOLOv8 output and applying NMS
    - Printing detections in the same schema used by src/inference.py

Usage:
    python serving/client_example.py \
        --triton-url localhost:8001 \
        --model weld_defect \
        --image weld.jpg \
        --conf 0.25 \
        --iou 0.45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFECT_LABELS = {
    0: "Crack",
    1: "Porosity",
    2: "Spatter",
    3: "Undercut",
    4: "Overlap",
}


def preprocess(image_path: Path, img_size: int = 640):
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]
    scale = img_size / max(h, w)
    new_h = round(h * scale)
    new_w = round(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    tensor = canvas.astype(np.float16).transpose(2, 0, 1)[None] / 255.0
    return tensor, scale, (h, w)


def nms(boxes, scores, iou_threshold: float):
    """Standard NMS. boxes in xyxy."""
    import numpy as np

    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess(raw_output, scale: float, original_shape, conf_threshold: float, iou_threshold: float):
    """YOLOv8 head output [1, 4+nc, 8400] -> detections list."""
    import numpy as np

    # raw_output shape: (1, 4 + num_classes, 8400)
    preds = raw_output[0].astype(np.float32)  # (4+nc, 8400)
    preds = preds.transpose(1, 0)  # (8400, 4+nc)

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    num_classes = class_scores.shape[1]

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    keep_mask = confidences >= conf_threshold
    if not keep_mask.any():
        return []

    boxes_xywh = boxes_xywh[keep_mask]
    class_ids = class_ids[keep_mask]
    confidences = confidences[keep_mask]

    # xywh -> xyxy
    x_c, y_c, w, h = (
        boxes_xywh[:, 0],
        boxes_xywh[:, 1],
        boxes_xywh[:, 2],
        boxes_xywh[:, 3],
    )
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    detections = []
    for cls_id in range(num_classes):
        cls_mask = class_ids == cls_id
        if not cls_mask.any():
            continue
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]
        keep = nms(cls_boxes, cls_scores, iou_threshold)
        for idx in keep:
            xx1, yy1, xx2, yy2 = cls_boxes[idx] / scale
            xx1 = max(0.0, float(xx1))
            yy1 = max(0.0, float(yy1))
            xx2 = min(float(original_shape[1]), float(xx2))
            yy2 = min(float(original_shape[0]), float(yy2))
            detections.append(
                {
                    "class_id": int(cls_id),
                    "class_name": DEFECT_LABELS.get(int(cls_id), str(cls_id)),
                    "confidence": float(cls_scores[idx]),
                    "bbox": [xx1, yy1, xx2, yy2],
                }
            )
    return detections


def run_inference(triton_url: str, model_name: str, image_path: Path, conf: float, iou: float):
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        sys.stderr.write(
            "tritonclient is not installed. Install with: "
            "pip install tritonclient[all]\n"
        )
        raise

    tensor, scale, original_shape = preprocess(image_path)

    client = grpcclient.InferenceServerClient(url=triton_url)
    if not client.is_server_ready():
        raise RuntimeError(f"Triton server not ready at {triton_url}")
    if not client.is_model_ready(model_name):
        raise RuntimeError(f"Model {model_name} not ready on Triton.")

    inputs = [grpcclient.InferInput("images", tensor.shape, "FP16")]
    inputs[0].set_data_from_numpy(tensor)

    outputs = [grpcclient.InferRequestedOutput("output0")]

    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    raw = response.as_numpy("output0")

    return postprocess(raw, scale, original_shape, conf, iou)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-url", type=str, default="localhost:8001")
    parser.add_argument("--model", type=str, default="weld_defect")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    detections = run_inference(
        triton_url=args.triton_url,
        model_name=args.model,
        image_path=args.image,
        conf=args.conf,
        iou=args.iou,
    )

    print(f"Detections ({len(detections)}):")
    for det in detections:
        print(
            f"  {det['class_name']:>10}  conf={det['confidence']:.2f}  "
            f"bbox=[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
            f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
