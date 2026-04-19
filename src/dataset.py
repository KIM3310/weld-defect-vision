"""Dataset preparation and validation for YOLO format weld defect data."""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def validate_dataset(data_yaml: Path) -> dict:
    """Validate YOLO dataset structure and return statistics.

    Checks:
    - Directory structure (images/labels for train/val/test)
    - Label format (class x_center y_center width height)
    - Class distribution
    - Image-label pairing
    """
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    base_path = data_yaml.parent
    stats = {"total_images": 0, "total_labels": 0, "class_counts": {}, "splits": {}}

    for split in ["train", "val", "test"]:
        img_dir = base_path / config.get(split, f"images/{split}")
        label_dir = base_path / "labels" / split

        if not img_dir.exists():
            stats["splits"][split] = {"status": "missing", "images": 0, "labels": 0}
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []

        split_classes: dict[int, int] = {}
        orphaned = 0

        for label_file in labels:
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        split_classes[cls] = split_classes.get(cls, 0) + 1

            img_stem = label_file.stem
            has_image = any(
                (img_dir / f"{img_stem}{ext}").exists() for ext in [".jpg", ".png"]
            )
            if not has_image:
                orphaned += 1

        stats["splits"][split] = {
            "status": "ok",
            "images": len(images),
            "labels": len(labels),
            "orphaned_labels": orphaned,
            "class_distribution": split_classes,
        }
        stats["total_images"] += len(images)
        stats["total_labels"] += len(labels)

        for cls, count in split_classes.items():
            stats["class_counts"][cls] = stats["class_counts"].get(cls, 0) + count

    return stats


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """Split a flat image/label directory into train/val/test splits.

    Args:
        images_dir: Directory containing all images
        labels_dir: Directory containing all YOLO-format label files
        output_dir: Base directory for split output
        train_ratio: Fraction of data for training
        val_ratio: Fraction for validation (rest goes to test)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    image_files = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    )
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train : n_train + n_val],
        "test": image_files[n_train + n_val :],
    }

    counts = {}
    for split_name, files in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            shutil.copy2(img_path, img_out / img_path.name)
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, lbl_out / label_path.name)

        counts[split_name] = len(files)

    return counts


def create_synthetic_dataset(
    output_dir: Path, n_images: int = 200, seed: int = 42
) -> None:
    """Generate synthetic weld defect images with YOLO annotations for pipeline testing.

    Creates images with colored shapes simulating different defect types.
    """
    random.seed(seed)
    np.random.seed(seed)

    defect_configs = {
        0: {"name": "crack", "color": (0, 0, 200), "shape": "line"},
        1: {"name": "porosity", "color": (0, 200, 0), "shape": "circles"},
        2: {"name": "spatter", "color": (200, 0, 0), "shape": "dots"},
        3: {"name": "undercut", "color": (0, 165, 255), "shape": "groove"},
        4: {"name": "overlap", "color": (128, 0, 128), "shape": "blob"},
    }

    for split in ["train", "val", "test"]:
        ratio = {"train": 0.7, "val": 0.15, "test": 0.15}[split]
        n = max(1, int(n_images * ratio))

        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            h, w = 640, 640
            img = np.full((h, w, 3), (80, 80, 80), dtype=np.uint8)

            # Weld bead background
            cv2.rectangle(img, (200, 280), (440, 360), (120, 120, 120), -1)

            annotations = []
            n_defects = random.randint(1, 3)

            for _ in range(n_defects):
                cls = random.randint(0, 4)
                cfg = defect_configs[cls]

                cx = random.randint(220, 420)
                cy = random.randint(290, 350)
                bw = random.randint(30, 80)
                bh = random.randint(20, 50)

                x1, y1 = cx - bw // 2, cy - bh // 2
                x2, y2 = cx + bw // 2, cy + bh // 2

                if cfg["shape"] == "line":
                    cv2.line(img, (x1, cy), (x2, cy), cfg["color"], 2)
                elif cfg["shape"] == "circles":
                    for _ in range(3):
                        px = random.randint(x1, x2)
                        py = random.randint(y1, y2)
                        cv2.circle(
                            img, (px, py), random.randint(3, 8), cfg["color"], -1
                        )
                elif cfg["shape"] == "dots":
                    for _ in range(5):
                        px = random.randint(x1, x2)
                        py = random.randint(y1, y2)
                        cv2.circle(
                            img, (px, py), random.randint(1, 4), cfg["color"], -1
                        )
                elif cfg["shape"] == "groove":
                    cv2.rectangle(img, (x1, y1), (x2, y2), cfg["color"], 2)
                elif cfg["shape"] == "blob":
                    cv2.ellipse(
                        img, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, cfg["color"], -1
                    )

                # YOLO format: class x_center y_center width height (normalized)
                annotations.append(f"{cls} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")

            cv2.imwrite(str(img_dir / f"weld_{i:04d}.jpg"), img)
            with open(lbl_dir / f"weld_{i:04d}.txt", "w") as f:
                f.write("\n".join(annotations))

    print(f"Synthetic dataset created at: {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        create_synthetic_dataset(Path("data"), n)
    elif len(sys.argv) > 1:
        stats = validate_dataset(Path(sys.argv[1]))
        import json

        print(json.dumps(stats, indent=2))
    else:
        print("Usage:")
        print("  Validate: python -m src.dataset data/weld_defect.yaml")
        print("  Synthetic: python -m src.dataset --synthetic [n_images]")
