# Weld Defect Vision

Industrial weld defect detection using **YOLOv8 fine-tuning**. Detects 5 types of welding defects: Crack, Porosity, Spatter, Undercut, and Overlap. Features a complete pipeline from data preparation to model serving via REST API.

Technical review pack: [`docs/technical-review-pack.md`](docs/technical-review-pack.md)

## Architecture

```
Input Image (640x640)
    │
    ▼
┌──────────────────────────┐
│  YOLOv8 Backbone (CSPNet) │  ← COCO pretrained weights
│  + FPN Neck               │
└──────────┬───────────────┘
           │ Multi-scale features
           ▼
┌──────────────────────────┐
│  Detection Head           │
│  3 scales: P3/P4/P5       │
│  Per anchor: bbox + cls   │
└──────────┬───────────────┘
           │
           ▼
    NMS → Detections
    [bbox, class, confidence]
```

## Key Features

- **YOLOv8 Fine-tuning**: Transfer learning from COCO-pretrained weights for weld defect domain
- **5 Defect Classes**: Crack, Porosity, Spatter, Undercut, Overlap
- **Data Augmentation**: Mosaic, MixUp, HSV jitter, flip (built into YOLO pipeline)
- **Evaluation**: mAP@50, mAP@50-95, per-class precision/recall charts
- **Visualization**: Bounding box overlay with class-specific colors and confidence scores
- **Inference API**: FastAPI endpoint for real-time detection + annotated image response
- **Docker**: GPU-enabled training and CPU/GPU serving containers

## Project Structure

```
weld-defect-vision/
├── src/
│   ├── config.py        # Hyperparameters, class labels, colors
│   ├── dataset.py       # YOLO dataset prep, validation, synthetic data
│   ├── train.py         # YOLOv8 training pipeline
│   ├── evaluate.py      # mAP evaluation with per-class charts
│   ├── inference.py     # Detection wrapper (single/batch)
│   └── visualize.py     # Bounding box drawing, detection grid
├── api/
│   ├── main.py          # FastAPI detection server
│   └── schemas.py       # Request/response models
├── data/
│   └── weld_defect.yaml # YOLO dataset configuration
├── tests/
│   ├── test_dataset.py  # Dataset and label format tests
│   ├── test_train.py    # Config and training component tests
│   └── test_api.py      # API endpoint tests
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Quick Start

### Setup

```bash
pip install -r requirements.txt
```

### Prepare Dataset

```bash
# Option 1: Use real weld defect dataset (place in data/ with YOLO format)
# Recommended: https://www.kaggle.com/datasets (search "weld defect detection")

# Option 2: Generate synthetic data for pipeline testing
python -m src.dataset --synthetic 200

# Validate dataset structure
python -m src.dataset data/weld_defect.yaml
```

Expected YOLO format:
```
data/
├── weld_defect.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/          # class x_center y_center width height (normalized)
    ├── train/
    ├── val/
    └── test/
```

### Train

```bash
python -m src.train
```

Outputs:
- `checkpoints/best.pt` — Best model weights
- `runs/detect/weld_defect/` — Training curves, PR curves, confusion matrix

### Evaluate

```bash
python -m src.evaluate checkpoints/best.pt
```

Outputs:
- `outputs/evaluation_results.json` — Full metrics (mAP, precision, recall per class)
- `outputs/per_class_metrics.png` — Per-class AP/P/R bar chart

### Inference API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Detect defects (JSON response)
curl -X POST http://localhost:8000/detect -F "file=@weld_image.jpg"

# Detect + visualize (annotated image response)
curl -X POST http://localhost:8000/detect/visualize -F "file=@weld_image.jpg" -o result.png
```

### Docker

```bash
# Inference server
docker compose up api

# Training (GPU)
docker compose --profile training run train
```

### Tests

```bash
pytest -v
```

## Defect Classes

| Class | Description | Color |
|-------|-------------|-------|
| Crack | Linear discontinuity in weld metal | Red |
| Porosity | Gas pockets trapped during solidification | Green |
| Spatter | Metal droplets expelled during welding | Blue |
| Undercut | Groove melted into base metal at weld toe | Orange |
| Overlap | Weld metal flowing over base metal without fusion | Purple |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch |
| Image Processing | OpenCV |
| Evaluation | matplotlib, seaborn |
| API | FastAPI, Uvicorn |
| Container | Docker, Docker Compose (GPU support) |
| Testing | pytest |
| Production Serving | Triton Inference Server (via `serving/`) |
| Edge Runtime | Jetson Orin + TensorRT (via `edge/`) |
| Industrial Integration | OPC-UA, MQTT, Kafka, REST webhook |

## Case Studies

See `docs/case-studies/` for composite customer narratives:

- [Shipyard pipeline](docs/case-studies/shipyard-pipeline.md) — 2-camera station integrated with PLC signals, Jetson Orin edge deployment, 90-day rollout with before/after metrics.
- [Automotive body shop](docs/case-studies/automotive-body-shop.md) — line-of-sight inspection at 1 weld / 2s throughput, integration with Andon systems.

Both are composite narratives with fabricated but plausible numbers, written in the SE-delivery format.

## Production Deployment

`docs/production/` covers the operational edge of a real deployment:

| Runbook | Topic |
|---------|-------|
| [edge-deployment.md](docs/production/edge-deployment.md) | ONNX/TensorRT export, Jetson deployment, INT8 quantization trade-offs |
| [model-serving.md](docs/production/model-serving.md) | Triton vs FastAPI vs ONNX Runtime Server; picking Triton |
| [mes-scada-integration.md](docs/production/mes-scada-integration.md) | OPC-UA, MQTT, PLC signal flow into the detection service |
| [monitoring-drift.md](docs/production/monitoring-drift.md) | Monitoring deployed models: confidence drift, re-evaluation cadence |
| [labeling-pipeline.md](docs/production/labeling-pipeline.md) | Active learning with human-in-the-loop label capture |

## Edge & Integration

- `edge/jetson-orin/` — Jetson-specific Dockerfile, systemd unit, setup script, watchdog.
- `serving/triton/` — Triton Inference Server model repository config for the weld_defect model.
- `integrations/` — reference integrations for OPC-UA, MQTT, Kafka, REST webhook.

## Model Governance

`governance/` follows industry-standard artifacts:

- [model-card.md](governance/model-card.md) — Google-style Model Card (intended use, training data, evaluation, limitations).
- [data-sheet.md](governance/data-sheet.md) — Gebru-style Datasheet for Datasets.
- [ethics-review.md](governance/ethics-review.md) — worker-monitoring implications, false-positive/negative cost framing.

## Benchmarks

`benchmarks/` contains reference benchmarks across devices and configurations. Sample results (see [benchmarks/results/](benchmarks/results/)):

| Device | Batch size | p50 latency (ms) | p95 latency (ms) | FPS |
|--------|-----------|-------------------|-------------------|-----|
| CPU (Xeon Gold 6138) | 1 | 420 | 480 | 2.4 |
| GPU (T4, PyTorch) | 1 | 38 | 52 | 26.3 |
| GPU (T4, TensorRT FP16) | 1 | 11 | 18 | 90.9 |
| Jetson Orin Nano (TensorRT INT8) | 1 | 22 | 31 | 45.4 |

Regenerate with `python benchmarks/latency_benchmark.py --devices cpu,gpu,jetson`.

## Related Projects

| Project | Relationship |
|---------|-------------|
| [retina-scan-ai](https://github.com/KIM3310/retina-scan-ai) | Sibling vision project — medical imaging classification with Grad-CAM |
| [AegisOps](https://github.com/KIM3310/AegisOps) | Operator handoff and incident analysis — directly applicable to plant floor escalation |
| [Nexus-Hive](https://github.com/KIM3310/Nexus-Hive) | Analytics layer consuming defect telemetry for trend analysis |
| [enterprise-llm-adoption-kit](https://github.com/KIM3310/enterprise-llm-adoption-kit) | Shared governance patterns (audit, RBAC) applicable to MES integration |
