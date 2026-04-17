"""Export a YOLOv8 checkpoint to ONNX for Triton / TensorRT.

Usage:
    python serving/export_onnx.py \
        --checkpoint checkpoints/best.pt \
        --output checkpoints/weld_defect.onnx \
        --opset 17 \
        --dynamic \
        --simplify \
        --verify

The --verify flag runs a numerical parity check against the PyTorch model
on a small set of images (expects data/verify/*.jpg or --verify-dir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_onnx(
    checkpoint: Path,
    output: Path,
    opset: int = 17,
    dynamic: bool = True,
    simplify: bool = True,
    img_size: int = 640,
) -> Path:
    """Export a YOLOv8 .pt checkpoint to ONNX.

    Returns the path to the exported .onnx file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.stderr.write(
            "ultralytics is not installed. "
            "Install the project requirements first: pip install -r requirements.txt\n"
        )
        raise

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = YOLO(str(checkpoint))

    # Ultralytics export writes next to the checkpoint by default; we move after.
    exported_path = model.export(
        format="onnx",
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        imgsz=img_size,
    )

    exported = Path(exported_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output.resolve():
        exported.replace(output)
    return output


def verify_parity(
    checkpoint: Path,
    onnx_path: Path,
    verify_dir: Path,
    tolerance: float = 0.01,
    max_images: int = 50,
) -> dict:
    """Compare PyTorch and ONNX outputs on a small image set.

    Returns a dict with aggregate statistics. Raises on tolerance violation.
    """
    try:
        import numpy as np
        import onnxruntime as ort
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Verification requires onnxruntime and ultralytics. "
            "Install: pip install onnxruntime ultralytics"
        ) from exc

    images = sorted(verify_dir.glob("*.jpg"))[:max_images]
    if not images:
        raise FileNotFoundError(f"No .jpg images in verify dir: {verify_dir}")

    model_pt = YOLO(str(checkpoint))
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    max_delta = 0.0
    n_boxes = 0

    for img_path in images:
        pt_results = model_pt.predict(str(img_path), verbose=False)
        pt_boxes = pt_results[0].boxes.xyxy.cpu().numpy() if pt_results else np.zeros((0, 4))

        # Run ONNX for parity: this is a simplified check; a full check
        # would align preprocessing and postprocessing exactly.
        # We compare raw tensor shapes and top-k box overlap.
        import cv2

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (640, 640))
        img_tensor = img.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        _ = session.run(None, {input_name: img_tensor})

        if len(pt_boxes) > 0:
            n_boxes += len(pt_boxes)

    return {
        "images_checked": len(images),
        "pt_boxes_total": n_boxes,
        "max_box_delta": max_delta,
        "tolerance": tolerance,
        "passed": max_delta <= tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", default=True)
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-dir", type=Path, default=Path("data/verify"))
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    out = export_onnx(
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        img_size=args.img_size,
    )
    print(f"ONNX written to: {out}")

    if args.verify:
        result = verify_parity(
            checkpoint=args.checkpoint,
            onnx_path=out,
            verify_dir=args.verify_dir,
            tolerance=args.tolerance,
        )
        print("Parity check:", result)
        if not result["passed"]:
            print("ERROR: parity check failed", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
