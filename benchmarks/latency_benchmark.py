"""Latency benchmark across batch sizes on the current device.

Measures p50/p95/p99 inference latency for the YOLOv8 weld-defect model
across a sweep of batch sizes. Auto-detects CPU / CUDA / Jetson and
reports device info alongside the timings.

Usage:
    python benchmarks/latency_benchmark.py \
        --model-path checkpoints/best.pt \
        --batch-sizes 1 2 4 8 16 \
        --warmup 50 \
        --iterations 500 \
        --img-size 640 \
        --output benchmarks/results/my-machine.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def detect_device() -> dict:
    """Return a dict describing the runtime device."""
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "has_cuda": False,
        "cuda_device_name": None,
        "is_jetson": False,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["has_cuda"] = bool(torch.cuda.is_available())
        if info["has_cuda"]:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
            info["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
    except ImportError:
        info["torch_version"] = None

    # Cheap Jetson probe: model file present on L4T.
    jetson_model = Path("/proc/device-tree/model")
    if jetson_model.exists():
        try:
            txt = jetson_model.read_text(errors="ignore").lower()
            info["is_jetson"] = "jetson" in txt or "nvidia" in txt
            info["jetson_model"] = txt.strip()
        except OSError:
            pass

    return info


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def bench_torch(
    model_path: Path,
    batch_size: int,
    img_size: int,
    warmup: int,
    iterations: int,
) -> dict:
    import numpy as np
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)

    dummy = np.random.rand(batch_size, 3, img_size, img_size).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy).to(device)

    # Warmup.
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.model(dummy_tensor)
    if device == "cuda":
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.model(dummy_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "batch_size": batch_size,
        "device": device,
        "iterations": iterations,
        "p50_ms": round(percentile(latencies_ms, 0.50), 3),
        "p95_ms": round(percentile(latencies_ms, 0.95), 3),
        "p99_ms": round(percentile(latencies_ms, 0.99), 3),
        "mean_ms": round(statistics.fmean(latencies_ms), 3),
        "stdev_ms": round(statistics.pstdev(latencies_ms), 3),
        "throughput_fps": round(
            batch_size / (statistics.fmean(latencies_ms) / 1000.0), 2
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device_info = detect_device()
    print(f"Device: {device_info.get('cuda_device_name') or 'CPU'}")
    if device_info["is_jetson"]:
        print(f"Jetson detected: {device_info.get('jetson_model')}")

    results: list[dict] = []
    for bs in args.batch_sizes:
        print(f"Benchmarking batch size {bs}...")
        try:
            row = bench_torch(
                model_path=args.model_path,
                batch_size=bs,
                img_size=args.img_size,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print(
                f"  p50={row['p50_ms']:.2f}  p95={row['p95_ms']:.2f}  "
                f"p99={row['p99_ms']:.2f}  throughput={row['throughput_fps']:.1f} fps"
            )
            results.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr)

    output_doc = {
        "device": device_info,
        "config": {
            "model_path": str(args.model_path),
            "img_size": args.img_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_doc, indent=2))
        print(f"Written: {args.output}")
    else:
        print(json.dumps(output_doc, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
