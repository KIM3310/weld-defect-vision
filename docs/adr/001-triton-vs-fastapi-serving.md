# ADR 001: Triton Inference Server for production serving, FastAPI for development

- **Status**: Accepted
- **Date**: 2026-03-20
- **Deciders**: ML owner, Plant IT owner (customer-side, shipyard deployment)

## Context

The repo ships a FastAPI-based inference server (`api/main.py`) that wraps a PyTorch-based YOLOv8 inference loop. This was adequate for development and for the shadow-mode pilot, but before cutover to production we needed to decide on the production serving stack.

Constraints:

- Jetson Orin AGX as the edge target; container deployment.
- Must support dynamic batching for the automotive use case (6 cameras firing roughly synchronously).
- Must support the existing model export (ONNX + TensorRT engine).
- Observability: Prometheus metrics, per-model latency histograms.
- Operational simplicity: plant IT is not ML-specialized.
- Model versioning: must support canary deployment and rollback.

## Decision

**Triton Inference Server** is the production serving stack. **FastAPI** remains available for development, CPU-only serving, and integration testing environments where Triton is over-specified.

## Options considered

### A. FastAPI + PyTorch (status quo)

Pros:
- Already built.
- Minimal ops surface.
- Python-only; trivial to extend.

Cons:
- No dynamic batching without custom code.
- No first-class model versioning.
- GPU utilization is tied to a single Python process's event loop.
- PyTorch runtime in production is 2-4x slower than the TensorRT engine.
- Prometheus metrics are manual.

### B. FastAPI + ONNX Runtime (in-process)

Pros:
- Better CPU throughput than PyTorch.
- Can consume the ONNX export pipeline we already have.
- Single-binary deployment.

Cons:
- Still no dynamic batching.
- Still no multi-model routing.
- Production path for Jetson TensorRT would be via ORT's TensorRT execution provider, which has been less stable than Triton's direct TensorRT integration.

### C. ONNX Runtime Server (standalone)

Pros:
- Purpose-built for ONNX.
- HTTP and gRPC.
- Supports ORT execution providers (TensorRT, OpenVINO, CUDA).

Cons:
- Slower-moving project than Triton.
- Dynamic batching is limited compared to Triton.
- No ensemble scheduler.
- Jetson support is viable but less polished than Triton's Jetson build.

### D. Triton Inference Server

Pros:
- First-class dynamic batching with `preferred_batch_size` and `max_queue_delay`.
- Model repository pattern with version policies (latest / specific / all), enabling canary deployment.
- Prometheus metrics out of the box.
- Ensemble scheduler for multi-model pipelines (applicable to the automotive Phase 2 spot-weld model).
- TensorRT execution is a first-class citizen.
- Jetson build is actively maintained.
- Community and documentation are strong.

Cons:
- `config.pbtxt` is protobuf text format, not YAML. Syntax errors can be confusing.
- Cold-start latency on large engines (10-40s on Orin).
- Operational surface is larger than FastAPI.

### E. TorchServe

Pros:
- Built for PyTorch.
- Model archival format (.mar) handles versioning.

Cons:
- Stewardship question (has been in maintenance mode for some time).
- Weaker Jetson story than Triton.
- TensorRT path is indirect.

## Rationale

Triton wins on the key production dimensions: dynamic batching (which is the automotive case's throughput multiplier), version-policy-based canary deployment, native TensorRT execution on Jetson, and Prometheus metrics. The operational cost of `config.pbtxt` and the cold-start time are acceptable given the value.

FastAPI is kept for development and for CPU-only environments because Triton's ceremony is not justified when the model and the hardware are small and the workload is dev-iteration.

## Consequences

### Positive

- Dynamic batching is a 2-3x throughput multiplier on multi-camera stations.
- Model versioning enables canary deployment without container rebuilds.
- Prometheus metrics are no-extra-work; plant ops dashboards are a Grafana configuration change.
- The ensemble scheduler unlocks the automotive Phase 2 spot-weld model without a capture-daemon rewrite.

### Negative

- Plant IT needs to learn Triton's model repository layout and `config.pbtxt` syntax. Onboarding time: ~2 days.
- Cold-start latency means rolling updates require health-check patience. Deploys use a 90s grace period before traffic shift.
- `config.pbtxt` syntax errors are caught at service start, not at CI time. We mitigate with a CI step that spins up Triton against the repo config to validate.

## References

- Production comparison: [`docs/production/model-serving.md`](../production/model-serving.md).
- Triton config: [`serving/triton/model_repository/weld_defect/config.pbtxt`](../../serving/triton/model_repository/weld_defect/config.pbtxt).
- Automotive case study (6-camera batch pattern): [`docs/case-studies/automotive-body-shop.md`](../case-studies/automotive-body-shop.md).
- Shipyard case study (edge Triton on Orin): [`docs/case-studies/shipyard-pipeline.md`](../case-studies/shipyard-pipeline.md).

## Revision history

- 2026-03-20: Initial decision.
- 2026-04-02: Added ensemble-scheduler motivation after automotive Phase 2 scoping.
