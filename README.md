# Weld Defect Vision

AI-powered welding defect detection and inspection reporting system for shipbuilding quality assurance.

Built as a demonstration of computer vision engineering capabilities for industrial NDT (Non-Destructive Testing) applications.

---

## Overview

Weld Defect Vision detects and classifies welding defects in images using a **ResNet-18 CNN backbone** with transfer learning, produces **ISO 5817 / AWS D1.1 aligned severity scores**, and generates structured **inspection reports** in JSON and HTML formats.

```
Upload weld image
      │
      ▼
┌─────────────────────┐
│  Preprocessing      │  CLAHE contrast enhancement, resize, EXIF correction
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  CNN Classifier     │  ResNet-18 + fine-tuned head → 7 defect classes
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Severity Scorer    │  0-100 score, ISO 5817 levels (Critical/High/Medium/Low)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Report Generator   │  JSON + HTML inspection report (IIW-style)
└─────────────────────┘
```

---

## Supported Defect Types

| Defect | ISO 6520-1 Group | Base Severity | Description |
|--------|-----------------|---------------|-------------|
| Crack | 1 | 90 | Linear discontinuity from thermal stress |
| Incomplete Fusion | 4 | 75 | Lack of fusion between weld and base metal |
| Undercut | 5 | 55 | Groove melted into base metal at weld toe |
| Porosity | 2 | 40 | Gas pockets trapped during solidification |
| Overlap | 5 | 30 | Weld metal protrusion beyond weld toe |
| Spatter | 6 | 15 | Metal particles expelled during welding |
| No Defect | — | 0 | Weld meets acceptance criteria |

---

## Project Structure

```
weld-defect-vision/
├── app/
│   ├── main.py                  # FastAPI application entry point
│   ├── models/
│   │   ├── classifier.py        # ResNet-18 CNN defect classifier
│   │   └── severity.py          # ISO 5817 severity scoring
│   ├── preprocessing/
│   │   └── pipeline.py          # CLAHE-based preprocessing pipeline
│   ├── reporting/
│   │   └── generator.py         # JSON + HTML inspection report generator
│   └── api/
│       └── routes.py            # FastAPI route definitions
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── tests/
│   ├── conftest.py              # Shared fixtures and synthetic image generators
│   ├── test_classifier.py       # Classifier unit tests
│   ├── test_preprocessing.py    # Pipeline unit tests
│   ├── test_severity.py         # Severity scoring tests
│   ├── test_api.py              # FastAPI endpoint tests
│   ├── test_reporting.py        # Report generation tests
│   └── test_production_quality.py  # End-to-end and edge case tests
├── data/sample/                 # Synthetic sample images
├── pyproject.toml
├── requirements.txt
└── Makefile
```

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

### Run Quality Checks

```bash
make quality-check   # lint + typecheck + tests
make lint            # ruff linter only
make typecheck       # mypy only
make test            # pytest only
make test-cov        # with coverage report
```

### Start the API Server

```bash
make run-api
# → http://localhost:8000/docs
```

### Start the Dashboard

```bash
make run-dashboard
# → http://localhost:8501
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health and model info |
| `GET` | `/ops/resource-pack` | Built-in inspection review pack |
| `GET` | `/ops/release-readiness` | Review-safe release checklist |
| `GET` | `/classes` | List all defect types with descriptions |
| `POST` | `/inspect` | Inspect a weld image → JSON result |
| `POST` | `/inspect/report` | Inspect a weld image → HTML report |
| `POST` | `/batch/inspect` | Batch inspection (up to 20 images) |
| `GET` | `/demo/synthetic` | Demo with synthetic weld image |

Interactive API docs available at `/docs` (Swagger UI) and `/redoc`.

## Reviewer Fast Path

1. `GET /api/v1/health`
2. `GET /api/v1/ops/resource-pack`
3. `GET /api/v1/ops/release-readiness`
4. `POST /api/v1/inspect`
5. `POST /api/v1/inspect/report`

### Example: Inspect an image

```bash
curl -X POST http://localhost:8000/api/v1/inspect \
  -F "file=@weld_image.jpg" \
  -F "weld_joint_id=J-2024-0042" \
  -F "inspector_notes=Visual inspection OK"
```

```json
{
  "report_id": "WDV-20240101-0001",
  "timestamp": "2024-01-01T09:00:00Z",
  "detection": {
    "defect_type": "porosity",
    "confidence": 0.847,
    "is_defect": true,
    "class_probabilities": { ... }
  },
  "severity": {
    "score": 38.5,
    "level": "medium",
    "recommended_action": "CAUTION – Document defect...",
    "is_acceptable": false
  },
  "conclusion": "MEDIUM: Porosity detected with 85% confidence..."
}
```

---

## Model Architecture

```
Input (224×224×3)
    │
    ▼
ResNet-18 Backbone (ImageNet pre-trained)
  - conv1, bn1, relu, maxpool  [frozen]
  - layer1, layer2             [frozen]
  - layer3, layer4             [fine-tuned]
    │
    ▼
Global Average Pooling → 512-dim
    │
    ▼
Classification Head:
  Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→7)
    │
    ▼
Softmax → 7-class probability distribution
```

### Demo Mode

When no trained checkpoint is available, the system activates **demo mode**, which uses image statistics (pixel intensity distribution, contrast, dark/bright pixel ratios) as heuristics for classification. This enables full pipeline demonstration and testing without training data or a GPU.

---

## Severity Scoring

Severity score (0–100) is computed as:

```
score = base_weight(defect_type) × √(confidence) + area_bonus + spatial_penalty
```

| Level | Score Range | Action |
|-------|-------------|--------|
| Critical | 75–100 | REJECT – immediate repair required |
| High | 50–74 | HOLD – additional NDE required |
| Medium | 25–49 | CAUTION – document and monitor |
| Low | 0–24 | PASS – within acceptance criteria |
| None | 0 | PASS – no defect detected |

Cracks always floor at Critical (≥75) per AWS D1.1 §6.12.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch 2.3 + torchvision |
| Image Processing | OpenCV (CLAHE), Pillow |
| API Server | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Linting | ruff |
| Type Checking | mypy |
| Testing | pytest + pytest-asyncio |
| Reporting | Jinja2 (HTML), JSON |

---

## Development

```bash
# Format code
make format

# Run tests with coverage
make test-cov

# Generate synthetic sample images
make generate-samples
```

---

## Roadmap

- [ ] YOLO-based bounding box detection for defect localisation
- [ ] Training pipeline with synthetic + augmented data
- [ ] DICOM / industrial image format support
- [ ] PostgreSQL-backed inspection database
- [ ] JWT authentication for API endpoints
- [ ] Export to PDF reports

---

*Developed for 한화오션 (Hanwha Ocean) AI Engineer application — demonstrating production-quality computer vision engineering for shipbuilding quality assurance.*
