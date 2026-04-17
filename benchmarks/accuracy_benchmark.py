"""Accuracy benchmark: per-class precision, recall, mAP, confusion matrix.

Uses the YOLOv8 validation mode under the hood and augments its output
with a 5x5 confusion matrix (one row/column per defect class) and a
per-class F1.

Usage:
    python benchmarks/accuracy_benchmark.py \
        --model-path checkpoints/best.pt \
        --data-yaml data/weld_defect.yaml \
        --split test \
        --iou 0.5 \
        --conf 0.001 \
        --output benchmarks/results/accuracy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFECT_CLASSES = ["crack", "porosity", "spatter", "undercut", "overlap"]


def run_validation(
    model_path: Path,
    data_yaml: Path,
    split: str,
    iou: float,
    conf: float,
    img_size: int,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        iou=iou,
        conf=conf,
        imgsz=img_size,
        plots=False,
        verbose=False,
    )

    per_class: list[dict] = []
    class_names = list(metrics.names.values()) if hasattr(metrics, "names") else DEFECT_CLASSES

    box = getattr(metrics, "box", None)
    if box is not None:
        p_arr = getattr(box, "p", [])
        r_arr = getattr(box, "r", [])
        ap50_arr = getattr(box, "ap50", [])
        ap_arr = getattr(box, "ap", [])

        for i, name in enumerate(class_names):
            p = float(p_arr[i]) if i < len(p_arr) else None
            r = float(r_arr[i]) if i < len(r_arr) else None
            ap50 = float(ap50_arr[i]) if i < len(ap50_arr) else None
            ap = float(ap_arr[i]) if i < len(ap_arr) else None
            f1 = None
            if p is not None and r is not None and (p + r) > 0:
                f1 = 2 * p * r / (p + r)
            per_class.append(
                {
                    "class_id": i,
                    "class_name": name,
                    "precision": round(p, 4) if p is not None else None,
                    "recall": round(r, 4) if r is not None else None,
                    "map50": round(ap50, 4) if ap50 is not None else None,
                    "map50_95": round(ap, 4) if ap is not None else None,
                    "f1": round(f1, 4) if f1 is not None else None,
                }
            )

    results_dict = {
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "split": split,
        "iou_threshold": iou,
        "conf_threshold": conf,
        "img_size": img_size,
        "aggregate": {
            "map50": float(metrics.box.map50) if box is not None else None,
            "map50_95": float(metrics.box.map) if box is not None else None,
            "precision": float(metrics.box.mp) if box is not None else None,
            "recall": float(metrics.box.mr) if box is not None else None,
        },
        "per_class": per_class,
    }

    cm = getattr(metrics, "confusion_matrix", None)
    if cm is not None and hasattr(cm, "matrix"):
        results_dict["confusion_matrix"] = cm.matrix.tolist()

    return results_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    doc = run_validation(
        model_path=args.model_path,
        data_yaml=args.data_yaml,
        split=args.split,
        iou=args.iou,
        conf=args.conf,
        img_size=args.img_size,
    )

    print(json.dumps(doc, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2))
        print(f"Written: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
