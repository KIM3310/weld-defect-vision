# Model Card: Weld Defect Vision (YOLOv8-based)

Following the Model Card structure introduced by Mitchell et al. 2019 and the subsequent Google Model Card Toolkit conventions. This card describes the model shipped from this repository; downstream deployments (e.g. the shipyard and automotive case studies) produce per-deployment model cards that inherit from this one and add deployment-specific context.

---

## Model details

- **Model name**: weld-defect-vision
- **Model version**: v0 (upstream)
- **Model type**: Object detection, single-stage anchor-free (YOLOv8)
- **Architecture**: YOLOv8 (Ultralytics). CSPNet backbone, PAN-FPN neck, decoupled detection head. Default variant: YOLOv8n (3.2M params); production variant: YOLOv8s (11.2M params).
- **Input**: 640x640 RGB image, uint8 / 0-255 per channel.
- **Output**: For each detected object: bounding box (xyxy), class label (one of 5), confidence score.
- **Framework**: PyTorch 2.2+. Exported to ONNX (opset 17) and TensorRT for production.
- **Developers**: Doeon Kim (https://github.com/KIM3310). This card describes the upstream reference model; each customer deployment has its own card with the customer-specific training data, hyperparameters, and evaluation.
- **Point of contact**: Via GitHub issues on the repo.
- **License**: Apache 2.0 for repo code; YOLOv8 / Ultralytics code under AGPL-3.0; consult Ultralytics license terms for production deployments that include the inference library.

### Model date

April 2026 (this card). Per-deployment training dates vary.

### Parent model

- **Parent**: YOLOv8 pretrained on COCO (Ultralytics `yolov8n.pt` or `yolov8s.pt`).
- **Fine-tuning**: Single-phase transfer learning on the 5-class weld defect dataset as configured in `src/train.py`.

---

## Intended use

### Primary intended uses

- **First-pass screening of surface-visible weld defects** in industrial welding QA workflows.
- **Augmenting a certified welding inspector (CWI)** by raising inspection coverage and shrinking time-to-flag.
- **Producing structured defect telemetry** for downstream analytics (defect rate by shift, by consumable lot, by operator).

### Primary intended users

- Welding inspectors (CWI-equivalent certified personnel in the target jurisdiction).
- Welding engineers responsible for WPS (welding procedure specification) adjustments.
- Plant QA managers and production engineers.
- ML/SE teams deploying the model at new customer sites.

### Out of scope uses

- **Replacing a certified welding inspector** in any jurisdiction where certified inspection is mandated (ISO 9606, AWS D1.1, ABS / KR / DNV class rules, etc.). The model is an augmentation layer, not a replacement.
- **Detection of subsurface defects**. The model sees surface only; porosity below the weld face, lack of fusion interior to the joint, and inclusions are outside of scope. Subsurface QA requires UT / RT / MT.
- **Primary safety decisions** where a single missed detection could cause loss of life. The model's recall is not high enough to be the sole layer for safety-critical decisions.
- **Performance evaluation of individual welders** without explicit labor / ethics architecture at the deploying site. See [`governance/ethics-architecture.md`](./ethics-architecture.md).
- **Detection of weld defect classes outside the 5 trained classes** (e.g. lack of fusion, burn-through, slag inclusion). These are outside the model's vocabulary and will either be ignored or misclassified.

---

## Factors

### Relevant factors

- **Welding process**: SAW, GMAW, GTAW, SMAW all produce different bead appearances. The upstream model is trained on a mix; customer deployments should include process-specific retraining data.
- **Consumable chemistry and lot**: wire chemistry affects bead color and sheen; a consumable lot change is a known drift driver.
- **Base material**: steel grade, surface finish (mill scale, blasted, painted, primer-coated) affect bead contrast.
- **Camera geometry**: distance, angle, illumination; the model is sensitive to these.
- **Ambient conditions**: temperature, dust, arc spatter residue.

### Evaluation factors

Production evaluation should stratify by:

- Welding process.
- Consumable lot (if available).
- Base material grade.
- Shift / operator cohort (with ethics architecture on per-operator analysis).

---

## Metrics

### Model performance measures

- **mAP@50**: mean average precision at IoU threshold 0.50 (primary aggregate metric).
- **mAP@50:95**: COCO-style averaged across 10 IoU thresholds 0.50-0.95.
- **Per-class precision / recall / F1** at a fixed confidence threshold.
- **Confusion matrix** with unmatched-detection and missed-object rows.

### Decision thresholds

The model outputs a continuous confidence. Per-class confidence thresholds are set at deployment time (see the shipyard case study for example values). The upstream repo does not bake in thresholds.

### Approaches to uncertainty and variability

- Per-class metrics are more informative than aggregate for imbalanced datasets.
- For production, compute per-class precision on CWI-confirmed detections and per-class recall against a weekly blind-relabeled holdout (see [`docs/production/labeling-pipeline.md`](../docs/production/labeling-pipeline.md)).
- Confidence scores are not calibrated probabilities; downstream thresholds should be set from the precision-recall curve at deployment.

---

## Evaluation data

### Upstream evaluation (reference)

Reference evaluation uses a split of the training dataset with:
- 80% train / 10% val / 10% test (held out during training).
- Random split; production deployments should use temporal or station-stratified splits to avoid leakage.

### Customer deployment evaluation

Each customer deployment has its own evaluation set:
- **Shipyard**: 960 labeled images from week 10-11 of capture (post-pilot), stratified across shifts.
- **Automotive**: 22,000 labeled ROIs, stratified by body model and camera station.

See the per-deployment model card (not included in this repo).

---

## Training data

### Datasets

The upstream reference model is trained on a mix of public weld-defect datasets and (for deployed models) customer-supplied data. Details and provenance are in [`governance/data-sheet.md`](./data-sheet.md).

Key characteristics:
- **Class imbalance**: Crack and Overlap are rare; Porosity and Spatter are common.
- **Variability**: lighting, camera geometry, base material varies across sources.
- **Labeling quality**: for customer deployments, inter-labeler Cohen's kappa is tracked and reported.

### Preprocessing

- Letterbox resize to 640x640 with 114 gray padding.
- Per-channel normalization is NOT applied (YOLOv8 uses 0-255 / 255 scaling).
- Augmentation at training time: mosaic, mixup, HSV jitter, random horizontal/vertical flip (see `src/config.py`).

---

## Quantitative analyses

### Unitary results (reference YOLOv8s)

| Class | mAP@50 | Precision | Recall |
|---|---|---|---|
| Crack | 0.81 | 0.78 | 0.72 |
| Porosity | 0.89 | 0.86 | 0.88 |
| Spatter | 0.92 | 0.93 | 0.91 |
| Undercut | 0.76 | 0.74 | 0.70 |
| Overlap | 0.68 | 0.71 | 0.63 |
| **Aggregate** | **0.81** | **0.80** | **0.77** |

Representative of the shipyard training run; actual numbers for your deployment will differ.

### Intersectional results

- **Class x precision (INT8)**: see [`benchmarks/results/class-balance-impact.json`](../benchmarks/results/class-balance-impact.json).
- **Precision x hardware target**: see [`benchmarks/results/cpu-vs-gpu-vs-jetson.json`](../benchmarks/results/cpu-vs-gpu-vs-jetson.json).
- **INT8 vs FP16 vs FP32 regression**: see [`docs/production/edge-deployment.md`](../docs/production/edge-deployment.md).

---

## Ethical considerations

See [`governance/ethics-architecture.md`](./ethics-architecture.md) for the full architecture. In summary:

1. **Labor implications**: system is framed as augmenting CWIs, not replacing them. Per-operator performance reporting requires explicit labor architecture at the deployment site.
2. **False-positive cost vs false-negative cost**: decision thresholds are tradeoffs with real safety and economic consequences. Escalation policy is set collaboratively with the customer's QA and welding engineering leads.
3. **Data consent**: weld images are not PII in most interpretations, but body-shop images may contain VIN stickers; these are redacted.
4. **Use in surveillance contexts**: out of scope. Deployments that could be used to penalize operators require labor engagement.

---

## Caveats and recommendations

- **The upstream model is a starting point**, not a production model. Plan for 2-6 weeks of customer-specific training.
- **Recall on Crack is the gating metric** for most industrial deployments; a model with aggregate mAP@50 of 0.85 but Crack recall of 0.60 is not production-ready.
- **INT8 regression is class-dependent**. If your gating class loses more than 2 mAP@50 points under INT8, use FP16.
- **Drift monitoring is not optional**. Consumable lot changes, new welders, new body models all produce drift. See [`docs/production/monitoring-drift.md`](../docs/production/monitoring-drift.md).
- **Do not skip the Gauge R&R**. The model must be compared to the human-human variability it is replacing/augmenting.

---

## References

- Mitchell et al., "Model Cards for Model Reporting", FAT* 2019.
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- ISO 5817 (Welding - Fusion-welded joints in steel, nickel, titanium and their alloys - Quality levels for imperfections) — the reference standard for weld imperfection classification.
- AWS D1.1 / D1.5 (American Welding Society structural codes).

---

## Cross-references in this repo

- Data sheet: [`governance/data-sheet.md`](./data-sheet.md).
- Ethics architecture: [`governance/ethics-architecture.md`](./ethics-architecture.md).
- Training code: [`src/train.py`](../src/train.py).
- Evaluation code: [`src/evaluate.py`](../src/evaluate.py).
- Accuracy benchmark: [`benchmarks/accuracy_benchmark.py`](../benchmarks/accuracy_benchmark.py).
- Case studies: [`docs/case-studies/`](../docs/case-studies/).
