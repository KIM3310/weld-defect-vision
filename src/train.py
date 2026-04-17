"""YOLOv8 training pipeline for weld defect detection."""

import json
from pathlib import Path

from ultralytics import YOLO

from src.config import TrainConfig


def train(config: TrainConfig | None = None) -> Path:
    """Train YOLOv8 on weld defect dataset.

    Fine-tunes a pretrained YOLOv8 model on the weld defect dataset
    with configured augmentation and hyperparameters.

    Returns path to the best checkpoint.
    """
    if config is None:
        config = TrainConfig()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(config.base_model)

    results = model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.img_size,
        lr0=config.learning_rate,
        patience=config.patience,
        workers=config.workers,
        seed=config.seed,
        project=str(config.output_dir),
        name="weld_defect",
        exist_ok=True,
        # Augmentation
        augment=config.augment,
        mosaic=config.mosaic,
        mixup=config.mixup,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        flipud=config.flipud,
        fliplr=config.fliplr,
        # Output
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    best_path = Path(config.output_dir) / "weld_defect" / "weights" / "best.pt"

    if best_path.exists():
        import shutil
        dst = config.checkpoint_dir / "best.pt"
        shutil.copy2(best_path, dst)
        print(f"\nBest model copied to: {dst}")

    # Save training summary
    summary = {
        "epochs_completed": results.epoch if hasattr(results, "epoch") else config.epochs,
        "best_model": str(best_path),
        "base_model": config.base_model,
        "img_size": config.img_size,
        "batch_size": config.batch_size,
    }

    summary_path = config.output_dir / "weld_defect" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return best_path


if __name__ == "__main__":
    train()
