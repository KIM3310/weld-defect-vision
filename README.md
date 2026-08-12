# Weld Defect Vision

## Live Demo

- [Open the public Cloudflare Pages demo](https://weld-defect-vision.pages.dev/)
- Scope: credential-free, synthetic-data demo for industrial validation discovery and quality evaluators.

Industrial weld defect detection research sandbox using **YOLOv8 fine-tuning**. It demonstrates 5 defect labels: Crack, Porosity, Spatter, Undercut, and Overlap. The repository is a synthetic-data validation discovery surface, not evidence of plant-floor performance.

Technical review pack: [`docs/architecture-pack.md`](docs/architecture-pack.md)

## System Overview

An industrial inspection AI workflow that sells value through validation discovery, data-suitability review, and operator-readable evidence.

| Area | Details |
|---|---|
| Users | Manufacturing quality teams, welding inspection groups, industrial AI teams, and edge deployment quality reviewers. |
| System scope | YOLOv8 workflow, synthetic scenarios, Triton/Jetson notes, MES/SCADA integration framing, model governance, and technical review pack. |
| Operating boundary | Prototype outputs need human inspector review; production use, yield claims, and customer outcomes require site-specific validation and acceptance criteria. |
| Evaluation path | Inspect the model card, validation notes, serving docs, and deterministic sample outputs. |

## Evaluation Path

- **Start here:** Read the model card and validation notes before checking detections.
- **Local demo:** Start the API with `uvicorn api.main:app --host 0.0.0.0 --port 8000`, then test `/detect` or `/detect/visualize`.
- **Checks:** Run `pytest -v`; keep human inspector review explicit when presenting outputs.

## Service Launch Playbook

- [Service launch playbook](docs/service-launch-playbook.md) maps the repository to its product scope, operating gates, operating boundaries, and risk controls.

## Architecture Notes

- [Architecture guide](docs/architecture-evidence-map.md) summarizes the system scope, first files to inspect, runtime commands, and known boundaries.
- [Quality notes](docs/quality-gate.md) lists the local checks, CI surface, and release expectations for this repository.
- [Enterprise readiness notes](docs/enterprise-readiness.md) outlines security, data, operations, integration, and handoff expectations.

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

## Fictional Deployment Scenarios

The files under `docs/case-studies/` are architecture exercises, **not customer case studies**. The organizations, engagements, deployments, datasets, benchmarks, and outcomes are fictional; every numeric value is fabricated and unmeasured:

- [Shipyard scenario](docs/case-studies/shipyard-pipeline.md) — hypothetical 2-camera, PLC, and Jetson design.
- [Automotive body-shop scenario](docs/case-studies/automotive-body-shop.md) — hypothetical line-of-sight and Andon integration design.

Use them to review requirements and failure modes only. Do not cite them as customer, model, hardware, or business evidence.

## Production Deployment

`docs/production/` covers deployment considerations that would need site-specific validation before use:

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
- [Ethics review](governance/ethics-architecture.md) — worker-monitoring implications, false-positive/negative cost framing.

## Benchmarks

`benchmarks/` contains runnable latency and accuracy harnesses. The committed files under [`benchmarks/results/`](benchmarks/results/) are explicitly labeled hand-authored fictional fixtures; they are not runner output or performance evidence.

Generate a measured local latency report only when you have a real checkpoint and can record the environment:

```bash
python benchmarks/latency_benchmark.py \
  --model-path checkpoints/best.pt \
  --batch-sizes 1 2 4 8 16 \
  --output benchmarks/results/my-machine-latency.json
```

See [`benchmarks/README.md`](benchmarks/README.md) for the evidence boundary and accuracy command.

## Related Projects

| Project | Relationship |
|---------|-------------|
| [retina-scan-ai](https://github.com/KIM3310/retina-scan-ai) | Sibling vision project — medical imaging classification with Grad-CAM |
| [AegisOps](https://github.com/KIM3310/AegisOps) | Operator handoff and incident analysis — directly applicable to plant floor escalation |
| [Nexus-Hive](https://github.com/KIM3310/Nexus-Hive) | Analytics layer consuming defect telemetry for trend analysis |
| [enterprise-llm-adoption-kit](https://github.com/KIM3310/enterprise-llm-adoption-kit) | Shared governance patterns (audit, RBAC) applicable to MES integration |

## Cloud + AI Architecture

- [Cloud + AI architecture blueprint](docs/cloud-ai-architecture.md)
- [Machine-readable architecture manifest](docs/architecture/blueprint.json)
- Validation command: `python3 scripts/validate_architecture_blueprint.py`

## Enterprise Productization

- [Product operating model](docs/product-operating-model.md) defines the product scope, trust boundary, operating checks, and service path for this repository.

## System Architecture

- [System architecture](docs/system-architecture.md) maps the runtime boundary, data/control flow, cloud or local deployment surface, and operating assumptions for this repository.

## Service Architecture

- [Service architecture](docs/service-architecture.md) defines the cloud resources, account information, cost controls, and production guardrails needed to turn this repo into a scoped service without publishing public financial assumptions.

<!-- search-growth-readme:start -->

## Search And Service Surface

- Public entry: free static inspection demo and architecture page
- Paid boundary: private industrial validation discovery for data suitability, baseline evaluation, model-card drafting, and human-review acceptance criteria
- Canonical URL: https://weld-defect-vision.pages.dev/
- Lead capture: https://kim3310-doeon-kim-portfolio.pages.dev/?offer=weld-defect-vision&inquiry=industrial-validation-discovery#private-inquiry
- Resource route: https://kim3310-doeon-kim-portfolio.pages.dev/resources/weld-defect-vision/
- Commercial route: https://kim3310-doeon-kim-portfolio.pages.dev/?offer=weld-defect-vision#service-offers
- CTA: Request private industrial validation discovery through the central inquiry URL
- Machine-readable offer: [docs/service-offer.json](docs/service-offer.json)
- Search growth implementation: [docs/search-growth-implementation.md](docs/search-growth-implementation.md)
- Revenue architecture: [docs/revenue-architecture.md](docs/revenue-architecture.md)

<!-- search-growth-readme:end -->

<!-- KIM3310:AD-DATA-PIVOT:START -->
## Free Resource, Advertising, and Aggregate Data

- [Public utility and architecture checklist](https://kim3310-doeon-kim-portfolio.pages.dev/resources/weld-defect-vision/)
- Revenue model: contextual advertising on the policy-eligible central resource page.
- Aggregate value: anonymous aggregate industrial vision validation interest and worksheet usage counts
- Boundary: ads allowed only on public validation resources; image uploads, inference results, defect records, and dashboards are ad-free
- Consent defaults off, DNT/GPC fail closed, and personal or sensitive data is never sold.
<!-- KIM3310:AD-DATA-PIVOT:END -->
