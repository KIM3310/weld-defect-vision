# ADR 002: Export to ONNX as a portable bridge; compile to TensorRT on the target device

- **Status**: Accepted
- **Date**: 2026-03-25
- **Deciders**: ML owner; input from NVIDIA solution architect (unaffiliated, informal architecture)

## Context

The reference model is trained in PyTorch (via Ultralytics YOLOv8). For production we need:

- GPU-accelerated inference on NVIDIA hardware (L4 in AWS, Jetson Orin at the edge).
- Sub-15 ms P95 per-image latency on Orin.
- Reproducible build artifacts for the plant change-control process.
- Portability across the dev / staging / prod hardware tiers without re-exporting from PyTorch each time.

## Decision

- **Export the PyTorch checkpoint to ONNX** as the canonical portable artifact.
- **Compile ONNX to a TensorRT engine** (`.plan`) on the target device for production inference.
- **Do not ship pre-compiled TensorRT engines** across device architectures or TensorRT major versions.

## Options considered

### A. Run PyTorch directly in production

Pros:
- No export step; fewer artifacts.

Cons:
- 2-4x slower than TensorRT INT8 on Jetson.
- Python runtime in production is not desired operationally.
- PyTorch versions must match exactly between dev and prod.

Rejected.

### B. Export to ONNX, serve ONNX Runtime (no TRT)

Pros:
- One artifact, runnable on multiple execution providers.
- Simpler build chain.

Cons:
- ORT-TensorRT execution provider exists but adds indirection; when it has parity problems with the underlying TRT version, debugging is painful.
- Direct TRT engine is measurably faster than ORT-via-TRT on Jetson (~15-25% in our benchmarks).
- For INT8 calibration, Triton's TRT path has a cleaner calibration-cache story than ORT-TRT.

Rejected for edge; acceptable fallback for dev.

### C. Export to ONNX, compile to TensorRT once, ship the .plan

Pros:
- Simplest for deployment: just copy the `.plan`.

Cons:
- TRT engines are NOT portable across GPU architectures (Orin vs L4 vs A100 all different).
- Not portable across TRT major versions.
- Moving JetPack versions on Orin requires recompile.

Rejected as primary path; used only for identical-device replica deployments.

### D. Export to ONNX, compile to TensorRT on target device at bootstrap time (chosen)

Pros:
- ONNX is the one portable artifact tracked in the model registry.
- The TRT engine is produced by the device at bootstrap and is correct-by-construction for that specific device.
- INT8 calibration cache is produced on-device with on-device calibration data, which is the recommended practice.
- Rolling to a new TRT version is a re-run of the compile step, not a model-registry change.

Cons:
- Bootstrap is longer (adds 2-10 minutes per device depending on hardware).
- Calibration cache lifecycle is an additional artifact to manage.

Chosen.

## Rationale

Treating ONNX as the portable artifact and TRT engines as device-specific derivatives maps cleanly onto the operational split: the ML team ships the ONNX; the plant IT / edge bootstrap flow produces the engine. The model registry stays simple (one ONNX per version) and rollouts across heterogeneous hardware work without branching artifacts.

## Consequences

### Positive

- Portable primary artifact (ONNX) simplifies the model registry.
- On-device compile catches device-specific issues at deploy time, not at serve time.
- INT8 calibration on-device with representative data is higher-quality than calibration on a bench machine.
- Supports the existing multi-tier deployment: dev on CPU (ORT), staging on L4 (TRT FP16), production on Orin (TRT INT8).

### Negative

- Bootstrap time is longer. Mitigation: pre-compile during the install window, not on hot path.
- Calibration data must be staged on the target device. Mitigation: seed from the plant's S3 to the edge NVMe during setup.
- Engine artifact is not checked into the repo; it is a build product. Requires discipline to not accidentally depend on a locally-built engine.

## Implementation

- [`serving/export_onnx.py`](../../serving/export_onnx.py): PyTorch -> ONNX with `--verify` numerical parity check.
- [`serving/export_tensorrt.py`](../../serving/export_tensorrt.py): ONNX -> TRT engine with INT8 calibration.
- [`edge/jetson-orin/setup.sh`](../../edge/jetson-orin/setup.sh): bootstrap step that calls export_tensorrt.py on the device.

## Accuracy tradeoffs

| Precision | Aggregate mAP@50 | Crack mAP@50 | Orin P95 |
|---|---|---|---|
| FP32 | 0.81 | 0.81 | 49 ms |
| FP16 | 0.81 | 0.81 | 27 ms |
| INT8 | 0.79 | 0.79 | 10 ms |

INT8 is the default for Orin edge deployments when the gating class mAP regression is acceptable. FP16 is used on customer-visible Class A surfaces (automotive Phase 1). See [`docs/production/edge-deployment.md`](../production/edge-deployment.md).

## References

- Export scripts: [`serving/export_onnx.py`](../../serving/export_onnx.py), [`serving/export_tensorrt.py`](../../serving/export_tensorrt.py).
- Edge deployment doc: [`docs/production/edge-deployment.md`](../production/edge-deployment.md).
- Model serving ADR: [`docs/adr/001-triton-vs-fastapi-serving.md`](./001-triton-vs-fastapi-serving.md).

## Revision history

- 2026-03-25: Initial decision.
- 2026-04-08: Added calibration-on-device detail after the first Orin bootstrap.
