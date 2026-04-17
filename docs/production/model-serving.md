# Model Serving: Triton vs FastAPI vs ONNX Runtime Server

The reference implementation in this repo uses two serving patterns:

- **FastAPI** for development and single-model CPU/GPU serving (see `api/main.py`).
- **Triton Inference Server** for production edge deployment (see `serving/triton/`).

This document compares the three main options we considered (Triton, FastAPI, ONNX Runtime Server), states the tradeoffs, and documents why Triton is the recommendation for production.

---

## 1. Summary table

| Dimension | FastAPI + PyTorch | ONNX Runtime Server | Triton Inference Server |
|---|---|---|---|
| Setup complexity | Low | Medium | Medium-high |
| Dynamic batching | No (manual) | Limited | Yes (first-class) |
| Multi-GPU / multi-instance | Manual | Limited | Yes (first-class) |
| Multi-model | Per-process | Per-process | Model repository |
| Hardware targets | CPU, CUDA | CPU, CUDA, TensorRT, OpenVINO | CPU, CUDA, TensorRT, ONNX RT, OpenVINO, DALI, custom |
| Protocols | HTTP/JSON | HTTP/JSON, gRPC | HTTP/JSON, gRPC, C API, SHM |
| Model ensembling | Manual | No | Yes (ensemble scheduler) |
| A/B / canary | Manual | Manual | Model version policy |
| Metrics (Prometheus) | Manual | Manual | Built-in |
| Jetson support | Yes | Yes | Yes (Jetson build) |
| License | Apache 2.0 (app code) | MIT | BSD-3 |
| Dev ergonomics | High | Medium | Lower (config-heavy) |

---

## 2. When to use FastAPI

The repo's `api/main.py` is FastAPI. This is the right choice for:

- **Development iteration.** Change a Python line, reload, test.
- **Single-model, moderate-throughput serving** where the cost of Triton's ceremony is not justified.
- **Edge CPU serving** when the hardware target has no CUDA or when the model is small.
- **Any environment where GPU is not available** (a customer's office, a cloud CPU instance for integration testing).

Not the right choice for:

- Multi-model deployments where you need to route per-request.
- Heterogeneous precision deployments (FP16 for model A, INT8 for model B, on the same GPU).
- Deployments requiring dynamic batching across concurrent clients.
- Jetson production where you want TensorRT throughput.

---

## 3. When to use ONNX Runtime Server

ORT Server has a smaller operational footprint than Triton. It is a reasonable fallback for edge deployments where Triton's feature surface is overkill. We evaluated it for the Jetson shipyard pilot and rejected for three reasons:

1. The ORT Server project has seen slower recent maintenance activity than Triton.
2. Dynamic batching is limited compared to Triton's scheduler.
3. We wanted the ensemble scheduler in Triton for the Phase 2 path where a second model (spot-weld classifier) would run alongside YOLOv8 and share preprocessing.

ORT in-process (using the Python or C++ ONNX Runtime library inside our own service process) is still a reasonable pattern for extremely constrained edge hardware, but we do not recommend running the ORT *Server* binary in production.

---

## 4. Why Triton for production

### 4.1 Dynamic batching is a throughput multiplier

In the automotive body shop case study, 6 cameras fire roughly simultaneously when a body enters the station. Without batching, 6 single-image requests each take ~9 ms; with batching we send 6 images as one batch taking ~22 ms. That is a 2.4x throughput improvement per GPU with one line of Triton config:

```
dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 10000
}
```

### 4.2 Model versioning and canary is operationally cheap

Triton's model repository layout is:

```
model_repository/
└── weld_defect/
    ├── config.pbtxt
    ├── 1/
    │   └── model.plan   (v1 TensorRT engine)
    ├── 2/
    │   └── model.plan   (v2 TensorRT engine)
    └── 3/
        └── model.plan   (v3 TensorRT engine)
```

The `version_policy` in `config.pbtxt` controls which versions are served. A rolling deploy is: write v3 to disk, adjust `version_policy` to serve both v2 and v3, shift traffic in the client, then retire v2. No container rebuild.

### 4.3 Built-in Prometheus metrics

Triton exposes `/metrics` with inference counts, latency histograms, queue depths, GPU utilization per model, and more. No extra instrumentation required for baseline observability.

### 4.4 The ensemble scheduler is future-proofing

When Phase 2 of the automotive case study adds a spot-weld regression model, the capture daemon does not need to orchestrate two model calls. Instead, an ensemble config can route camera-cropped ROIs to either model based on `class_policy` and return a single response. The capture daemon sees one Triton endpoint.

### 4.5 Common frustrations with Triton

- `config.pbtxt` is protobuf text format, not YAML. Small syntax errors fail load with cryptic messages.
- Jetson builds lag the server build by a few weeks; plan for version pinning.
- Cold-start on large engines can be slow (10-40 seconds per engine on Orin). The instance-group `count` controls warm-instance quantity; plan your startup budget.

---

## 5. Triton config walkthrough

`serving/triton/model_repository/weld_defect/config.pbtxt`:

```
name: "weld_defect"
platform: "tensorrt_plan"
max_batch_size: 16

input [
  {
    name: "images"
    data_type: TYPE_FP16
    dims: [3, 640, 640]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP16
    dims: [84, 8400]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 10000
}

version_policy: { latest: { num_versions: 1 } }
```

Notes:

- `platform: "tensorrt_plan"` tells Triton this directory contains a TensorRT engine. For ONNX without TRT compile, use `platform: "onnxruntime_onnx"`.
- `max_batch_size` gates dynamic batching; `dims` describes the per-sample shape (batch dim is implicit).
- Output dim `[84, 8400]` is YOLOv8's raw head output before NMS (4 box + 80 class scores if coco, or 4 + 5 classes for this model = 9 — re-check for custom class counts). For the 5-class weld model the output is `[9, 8400]`, not `[84, 8400]`; update accordingly per trained model.
- `instance_group.count: 2` lets Triton run two concurrent streams of the engine on the same GPU, which improves utilization when the batch sizes are small.
- `version_policy latest num_versions: 1` serves only the newest version; change to `specific: { versions: [2, 3] }` for canary.

---

## 6. Client patterns

For a thin client we use the Triton Python client library (`tritonclient[all]`). See [`serving/client_example.py`](../../serving/client_example.py). Key points:

- Prefer gRPC over HTTP for lower per-request overhead.
- Use `SharedMemoryRegion` for zero-copy on same-host clients (optional; improves latency by 2-4 ms on Orin).
- Post-process NMS client-side using the YOLOv8 post-processing; this keeps the engine shape-stable.

---

## 7. Observability

| Signal | Source | Alert threshold |
|---|---|---|
| Inference p99 latency | Triton metrics | > 25 ms for 5 min |
| Queue depth | Triton metrics | > 8 for 2 min |
| GPU utilization | `nvidia-smi` or DCGM | < 20% for 10 min (under-utilized; consider scaling down) |
| Model load errors | Triton log | any |
| Request error rate | Triton metrics | > 0.5% over 1 min |

Prometheus scrapes Triton at `:8002/metrics`. Grafana dashboard lives under [plant ops repo, not included here].

---

## 8. References

- Triton Inference Server documentation: `https://github.com/triton-inference-server/server`.
- YOLOv8 export: `https://docs.ultralytics.com/modes/export/`.
- ONNX Runtime: `https://onnxruntime.ai/`.
- ADR on this decision: [`docs/adr/001-triton-vs-fastapi-serving.md`](../adr/001-triton-vs-fastapi-serving.md).
- Automotive case study (6-camera batch pattern): [`docs/case-studies/automotive-body-shop.md`](../case-studies/automotive-body-shop.md).
- Shipyard case study (single-camera edge): [`docs/case-studies/shipyard-pipeline.md`](../case-studies/shipyard-pipeline.md).
