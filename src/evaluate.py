"""Evaluation pipeline for weld defect detection model."""

import json
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.config import DEFECT_LABELS, TrainConfig


class ClassMetrics(TypedDict):
    precision: float
    recall: float
    ap50: float
    ap: float


class EvaluationMetrics(TypedDict):
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    per_class: dict[str, ClassMetrics]


def evaluate(
    checkpoint_path: str | Path, config: TrainConfig | None = None
) -> EvaluationMetrics:
    """Evaluate YOLOv8 model on test set.

    Computes mAP@50, mAP@50-95, per-class precision/recall, and generates
    visualization plots.

    Returns dict with all evaluation metrics.
    """
    if config is None:
        config = TrainConfig()

    model = YOLO(str(checkpoint_path))

    results = model.val(
        data=str(config.data_yaml),
        split="test",
        imgsz=config.img_size,
        batch=config.batch_size,
        conf=config.conf_threshold,
        iou=config.iou_threshold,
        plots=True,
        save_json=True,
        verbose=True,
    )

    per_class: dict[str, ClassMetrics] = {}
    metrics: EvaluationMetrics = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class": per_class,
    }

    if (
        hasattr(results.box, "ap_class_index")
        and results.box.ap_class_index is not None
    ):
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = DEFECT_LABELS.get(int(cls_idx), f"class_{cls_idx}")
            per_class[cls_name] = {
                "precision": float(results.box.p[i]) if i < len(results.box.p) else 0.0,
                "recall": float(results.box.r[i]) if i < len(results.box.r) else 0.0,
                "ap50": (
                    float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0
                ),
                "ap": float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
            }

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_metrics_bar(per_class, output_dir / "per_class_metrics.png")

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*50}")
    print(f"mAP@50:    {metrics['mAP50']:.4f}")
    print(f"mAP@50-95: {metrics['mAP50_95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"{'='*50}")

    for cls_name, cls_metrics in per_class.items():
        print(
            f"  {cls_name}: AP50={cls_metrics['ap50']:.3f} P={cls_metrics['precision']:.3f} R={cls_metrics['recall']:.3f}"
        )

    print(f"\nResults saved: {results_path}")

    return metrics


def _plot_metrics_bar(per_class: dict[str, ClassMetrics], output_path: Path) -> None:
    """Generate per-class metrics bar chart."""
    if not per_class:
        return

    classes = list(per_class.keys())
    ap50 = [per_class[c]["ap50"] for c in classes]
    precision = [per_class[c]["precision"] for c in classes]
    recall = [per_class[c]["recall"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    _fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, ap50, width, label="AP@50", color="#2196F3")
    ax.bar(x, precision, width, label="Precision", color="#4CAF50")
    ax.bar(x + width, recall, width, label="Recall", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Detection Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class metrics chart saved: {output_path}")


if __name__ == "__main__":
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    evaluate(ckpt)
