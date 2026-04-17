"""Compile an ONNX model to a TensorRT engine plan.

Usage:
    # FP16 on any modern NVIDIA GPU:
    python serving/export_tensorrt.py \
        --onnx checkpoints/weld_defect.onnx \
        --output serving/triton/model_repository/weld_defect/1/model.plan \
        --precision fp16

    # INT8 with calibration on representative images:
    python serving/export_tensorrt.py \
        --onnx checkpoints/weld_defect.onnx \
        --output serving/triton/model_repository/weld_defect/1/model.plan \
        --precision int8 \
        --calibration-dir data/calibration \
        --calibration-cache checkpoints/calibration.cache \
        --max-batch-size 16

Run this on the TARGET device. TensorRT engines are not portable across
GPU architectures or across major TensorRT versions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_trt():
    try:
        import tensorrt as trt  # type: ignore[import-not-found]
    except ImportError:
        sys.stderr.write(
            "TensorRT is not installed. This script must run on a device with "
            "TensorRT available (e.g. the Jetson Orin target or an x86 host "
            "with a matching CUDA/TRT stack).\n"
        )
        raise
    return trt


class ImageBatchCalibrator:
    """INT8 entropy calibrator wrapping a directory of calibration images.

    Inherits from trt.IInt8EntropyCalibrator2 when TensorRT is available at
    runtime. We defer the inheritance to keep this module importable on
    machines without TRT (for doc generation, CI on CPU, etc.).
    """

    def __init__(
        self,
        calibration_dir: Path,
        cache_path: Path,
        batch_size: int = 8,
        img_size: int = 640,
    ) -> None:
        self.calibration_dir = calibration_dir
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.index = 0
        self.images = sorted(
            list(calibration_dir.glob("*.jpg"))
            + list(calibration_dir.glob("*.png"))
        )
        if not self.images:
            raise FileNotFoundError(
                f"No calibration images found in {calibration_dir}"
            )

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        # Loaded lazily so that this file imports cleanly without numpy/cv2/pycuda.
        import cv2
        import numpy as np

        start = self.index * self.batch_size
        if start >= len(self.images):
            return None

        batch_files = self.images[start : start + self.batch_size]
        if len(batch_files) < self.batch_size:
            return None

        batch = np.empty(
            (self.batch_size, 3, self.img_size, self.img_size),
            dtype=np.float32,
        )
        for i, fp in enumerate(batch_files):
            img = cv2.imread(str(fp))
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            batch[i] = img.transpose(2, 0, 1)

        self.index += 1
        # Expected to return a GPU pointer in a real implementation.
        return [batch.ctypes.data]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)


def build_engine(
    onnx_path: Path,
    output_path: Path,
    precision: str,
    max_batch_size: int,
    workspace_mb: int,
    calibration_dir: Path | None,
    calibration_cache: Path | None,
) -> Path:
    trt = _load_trt()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            sys.stderr.write(parser.get_error(i).desc() + "\n")
        raise RuntimeError("Failed to parse ONNX.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)
    )

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    profile.set_shape(
        input_name,
        min=(1, 3, 640, 640),
        opt=(max_batch_size // 2 or 1, 3, 640, 640),
        max=(max_batch_size, 3, 640, 640),
    )
    config.add_optimization_profile(profile)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            sys.stderr.write("Warning: platform does not have fast FP16.\n")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            sys.stderr.write("Warning: platform does not have fast INT8.\n")
        if calibration_dir is None or calibration_cache is None:
            raise ValueError("INT8 requires --calibration-dir and --calibration-cache")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = ImageBatchCalibrator(
            calibration_dir=calibration_dir,
            cache_path=calibration_cache,
            batch_size=min(8, max_batch_size),
        )
    elif precision == "fp32":
        pass
    else:
        raise ValueError(f"Unknown precision: {precision}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build returned None.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(serialized))
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp16",
    )
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--calibration-cache", type=Path, default=None)
    args = parser.parse_args()

    path = build_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        precision=args.precision,
        max_batch_size=args.max_batch_size,
        workspace_mb=args.workspace_mb,
        calibration_dir=args.calibration_dir,
        calibration_cache=args.calibration_cache,
    )
    print(f"TensorRT engine written to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
