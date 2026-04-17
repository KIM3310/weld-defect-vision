# Serving

Production serving uses **NVIDIA Triton Inference Server**. FastAPI (`api/main.py`) remains available for development and CPU-only environments; the rationale is documented in [`docs/adr/001-triton-vs-fastapi-serving.md`](../docs/adr/001-triton-vs-fastapi-serving.md) and [`docs/production/model-serving.md`](../docs/production/model-serving.md).

This directory contains:

```
serving/
├── README.md                         (this file)
├── export_onnx.py                    PyTorch .pt -> ONNX
├── export_tensorrt.py                ONNX -> TensorRT plan (FP16 / INT8)
├── client_example.py                 Triton client making an inference call
└── triton/
    ├── docker-compose.yml            Triton server + client
    └── model_repository/
        └── weld_defect/
            ├── config.pbtxt          Triton model config
            └── 1/                    Model version directory (engine or ONNX goes here)
```

## Typical flow

```
checkpoints/best.pt
    │  python serving/export_onnx.py --checkpoint checkpoints/best.pt \
    │                                --output checkpoints/weld_defect.onnx
    ▼
checkpoints/weld_defect.onnx
    │  python serving/export_tensorrt.py --onnx checkpoints/weld_defect.onnx \
    │                                    --output serving/triton/model_repository/weld_defect/1/model.plan \
    │                                    --precision int8 \
    │                                    --calibration-dir data/calibration/
    ▼
serving/triton/model_repository/weld_defect/1/model.plan
    │  docker compose -f serving/triton/docker-compose.yml up triton
    ▼
Triton running on :8000 (HTTP), :8001 (gRPC), :8002 (metrics)
    │  python serving/client_example.py --image weld.jpg
    ▼
Detections printed to stdout
```

## Checking the config

Validate the config.pbtxt syntax before running Triton:

```bash
# Triton will refuse to start if config.pbtxt is malformed.
# A quick syntax check is to look at the server log on startup.
docker compose -f serving/triton/docker-compose.yml up triton 2>&1 | head -40
```

## Model versioning

Add version 2 by creating a `2/` directory alongside `1/`:

```
model_repository/weld_defect/
├── config.pbtxt
├── 1/
│   └── model.plan    (v1)
└── 2/
    └── model.plan    (v2, new)
```

The `version_policy` in `config.pbtxt` controls which versions are served. Default is `latest`; switch to `specific` for canary.

## Latency benchmarks

See [`benchmarks/latency_benchmark.py`](../benchmarks/latency_benchmark.py) for a script that sweeps batch size and reports p50 / p95 / p99. Sample results are in [`benchmarks/results/cpu-vs-gpu-vs-jetson.json`](../benchmarks/results/cpu-vs-gpu-vs-jetson.json).

## Export parity check

Before promoting a new engine, run:

```bash
python serving/export_onnx.py --checkpoint checkpoints/best.pt --output /tmp/weld.onnx --verify
```

The `--verify` flag runs a 50-image numerical comparison against the PyTorch original. Max acceptable IoU delta per box is 0.01.
