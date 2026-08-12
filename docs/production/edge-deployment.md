# Edge Deployment: Jetson Orin and TensorRT

> **Evidence boundary:** this is reference design guidance, not a deployment record. The linked shipyard scenario and its latency, accuracy, calibration, and thermal values are fictional and unmeasured. Verify device specifications and generate checkpoint-specific measurements on target hardware before setting an acceptance gate.

This document describes a possible export path (PyTorch → ONNX → TensorRT), INT8 calibration procedure, precision trade-off review, and runtime pattern for a Jetson Orin edge node.

The reference context is the fictional shipyard scenario in [`docs/case-studies/shipyard-pipeline.md`](../case-studies/shipyard-pipeline.md).

---

## 1. Target devices

| Device | Memory | Compute | Power budget | Typical latency |
|---|---|---|---|---|
| Jetson Orin AGX 64 GB | 64 GB | 275 TOPS (INT8) | 15-60 W | 8-12 ms (INT8) |
| Jetson Orin NX 16 GB | 16 GB | 100 TOPS (INT8) | 10-25 W | 14-22 ms (INT8) |
| Jetson Orin Nano 8 GB | 8 GB | 40 TOPS (INT8) | 7-15 W | 25-35 ms (INT8) |

For the weld-defect workload in the shipyard case study the Orin AGX 64 GB was chosen because the cell captures three images per weld event and the total inference budget per weld is ~40 ms. The Orin NX is viable if the trigger cadence is relaxed to one image per event.

---

## 2. Export path

```
checkpoints/best.pt (PyTorch, FP32)
         │
         │  serving/export_onnx.py
         ▼
     weld_defect.onnx  (ONNX, opset 17, FP32)
         │
         │  serving/export_tensorrt.py
         │    ├── calibration dataset (500 representative images)
         │    └── INT8 calibrator
         ▼
     weld_defect_int8.plan  (TensorRT engine, INT8)
         │
         │  copy to Orin /opt/triton/models/weld_defect/1/model.plan
         ▼
     Triton Inference Server serves the engine
```

### 2.1 ONNX export

The export is done with Ultralytics' built-in ONNX exporter wrapped in `serving/export_onnx.py`. Opset 17 is the baseline; the main requirements are:

- Dynamic axes on batch dimension: enables dynamic batching in Triton.
- No NMS inside the graph: the graph returns raw box tensor, NMS runs in client-side post-processing (this keeps the engine shape-stable and simplifies INT8 calibration).
- `simplify=True`: runs `onnx-simplifier` to fuse constants and remove redundant ops.

Verify the export numerically against the PyTorch original on a 50-image holdout. Compare per-box IoU. Max acceptable IoU delta: 0.01. Typical observed delta: 0.003.

### 2.2 TensorRT compilation

The ONNX graph is compiled to a TensorRT engine with `trtexec` or the Python API. For INT8 we supply a calibrator that iterates 500 representative images from the training set.

INT8 calibration pattern (entropy calibrator 2):

- Select 500 images that span the class distribution, camera positions, and environmental conditions of the production station.
- Run the calibrator on the target device (Orin). Calibration on the Orin itself produces a calibration cache that is trustworthy for that specific device.
- Calibration artifact (`calibration.cache`) is version-controlled alongside the model in the model registry.

Accuracy impact from INT8 (shipyard case study, per-class mAP@50):

| Class | FP32 | FP16 | INT8 | INT8 delta |
|---|---|---|---|---|
| Crack | 0.81 | 0.81 | 0.79 | -0.02 |
| Porosity | 0.89 | 0.89 | 0.87 | -0.02 |
| Spatter | 0.92 | 0.92 | 0.90 | -0.02 |
| Undercut | 0.76 | 0.76 | 0.74 | -0.02 |
| Overlap | 0.68 | 0.67 | 0.65 | -0.03 |
| **Aggregate** | **0.81** | **0.81** | **0.79** | **-0.02** |

Latency impact (Jetson Orin AGX, batch size 1, 640x640):

| Precision | Latency (ms, P95) | Memory (MB) |
|---|---|---|
| FP32 | 49 | 1,140 |
| FP16 | 27 | 610 |
| INT8 | 10 | 320 |

INT8 is the right choice when the mAP regression on the gating class (Crack for this domain) stays within spec. If a deployment needs tighter accuracy (automotive body shop Class A surfaces, see `docs/case-studies/automotive-body-shop.md`), FP16 is the reasonable fallback.

### 2.3 Engine portability

TensorRT engines are not portable across GPU architectures or across TRT versions. A plan file compiled on one Orin unit may be reused across identical Orin units of the same JetPack version, but any change in JetPack or any change to a different compute capability requires a recompile. We embed the compilation step in the device bootstrap flow rather than shipping a precompiled engine.

---

## 3. Runtime on Orin

### 3.1 OS and stack

- JetPack 6.1 (L4T r36.4, Ubuntu 22.04).
- CUDA 12.6, cuDNN 9.3, TensorRT 10.3.
- Triton Inference Server 25.01 (Jetson build).
- Python 3.10 for the capture daemon.

### 3.2 Services

Two systemd units manage the runtime:

- `triton.service` — Triton running with one model (weld_defect) and the TensorRT plan.
- `weld-defect.service` — the Python capture daemon that subscribes to the robot controller's OPC-UA tag, captures images on arc-off, calls Triton, and publishes results to MQTT.

See [`edge/jetson-orin/systemd/weld-defect.service`](../../edge/jetson-orin/systemd/weld-defect.service) for the systemd unit.

The capture daemon and Triton communicate via gRPC on localhost. This keeps the serialization cost low (shared-memory-ish via Unix domain sockets is an option; we chose gRPC for operational simplicity).

### 3.3 Power mode and thermal

The Orin has multiple power modes (`nvpmodel`). For this workload we use mode 0 (MAXN) during inference windows and mode 4 (50W) during idle. The switch is handled by the daemon when the weld cadence drops.

Thermal observations from the shipyard deployment:

- Orin tj (junction temperature) under sustained inference at ambient 25C: ~61C.
- At ambient 38C (summer factory): ~84C, which triggers throttling.
- Mitigation: active case fan and a 2-cm baffle deflecting exhaust away from intake. Held tj at 72C under the same load.

Monitor thermal with `tegrastats`:

```
sudo tegrastats --interval 1000
```

The capture daemon reads `tegrastats` output and publishes per-minute summary to MQTT topic `yardk/r7/edge/thermal`. See [`edge/common/watchdog.py`](../../edge/common/watchdog.py).

### 3.4 Storage

- Rootfs on the built-in eMMC (32 GB).
- Image archive and model artifacts on an external M.2 NVMe (1 TB). Image archive rolls on a 14-day window (older images are copied to the plant's S3-compatible store before deletion).
- Calibration cache and engine plan: `/opt/triton/models/weld_defect/1/model.plan` (NVMe).

### 3.5 Networking

Dual NIC: one PoE GigE for the camera, one 1 GbE for plant LAN (MQTT, MES, NTP, model-pull).

Static IP on the plant LAN, negotiated with plant IT. Orin is behind the plant firewall; no direct internet. Model artifacts are pulled from a plant-local registry (`artifactory.plant.internal`).

---

## 4. Failure modes and mitigations

| Failure mode | Detection | Mitigation |
|---|---|---|
| Triton crash (OOM, bad input, TRT internal error) | systemd watchdog + HTTP health check | Restart via systemd, page on 3 restarts in 10 min |
| Camera disconnect | GigE Vision heartbeat loss | Auto-reconnect, alarm via MQTT if disconnect > 30 s |
| Thermal throttle | tegrastats tj > 78C | Page ops; fall back to FP16 path if INT8 engine throttles below budget |
| Plan file corrupt | Load failure at service start | Re-compile from ONNX; if ONNX absent, pull from registry |
| Model drift | Confidence histogram monitor | See `monitoring-drift.md` |
| SSD wear | SMART error counter monitor | Swap to hot-spare device |
| Clock skew | NTP drift > 500 ms | Alarm; gate MQTT publishes until clock is within tolerance |

---

## 5. Deployment checklist

Before cutover, verify each:

- [ ] Model version deployed matches the version signed off by QA.
- [ ] Calibration cache produced on the target device is in use.
- [ ] Numerical parity with the PyTorch model confirmed on a 50-image holdout.
- [ ] P95 latency measured and within budget.
- [ ] Triton and capture daemon start on boot.
- [ ] MQTT publish confirmed; MES receives the test event.
- [ ] OPC-UA client confirms subscription to robot controller tag and receives pulse on test arc.
- [ ] Thermal tj under load measured below 75C at expected worst-case ambient.
- [ ] Fallback path (revert to previous engine) tested.
- [ ] Rollback runbook in plant ops manual.

---

## 6. References in this repo

- Export scripts: [`serving/export_onnx.py`](../../serving/export_onnx.py), [`serving/export_tensorrt.py`](../../serving/export_tensorrt.py).
- Triton config: [`serving/triton/model_repository/weld_defect/config.pbtxt`](../../serving/triton/model_repository/weld_defect/config.pbtxt).
- Jetson container: [`edge/jetson-orin/Dockerfile`](../../edge/jetson-orin/Dockerfile).
- systemd unit: [`edge/jetson-orin/systemd/weld-defect.service`](../../edge/jetson-orin/systemd/weld-defect.service).
- Watchdog: [`edge/common/watchdog.py`](../../edge/common/watchdog.py).
- Benchmarks: [`benchmarks/latency_benchmark.py`](../../benchmarks/latency_benchmark.py).
- ADR on edge vs cloud: [`docs/adr/003-edge-vs-cloud-inference.md`](../adr/003-edge-vs-cloud-inference.md).
- ADR on ONNX vs TensorRT: [`docs/adr/002-onnx-vs-tensorrt.md`](../adr/002-onnx-vs-tensorrt.md).
