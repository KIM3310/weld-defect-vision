# Datasheet for Datasets: Weld Defect Vision

Following Gebru et al. "Datasheets for Datasets" (2018, revised 2021). This datasheet describes the reference training dataset used by the upstream model. Customer deployments have per-deployment datasheets that describe their own data.

---

## Motivation

### Why was the dataset created?

To fine-tune a COCO-pretrained object-detection model (YOLOv8) for the specific domain of industrial weld-defect visual inspection, detecting five canonical surface-defect classes: Crack, Porosity, Spatter, Undercut, Overlap.

### For what tasks is it suitable?

- Object detection training (bounding box + class).
- Object detection evaluation (precision, recall, mAP).
- Dataset-distribution analysis (class balance, image diversity).
- Drift baseline computation for downstream deployments.

Not suitable for:
- Training models that require subsurface defect ground truth.
- Training models that require operator identity labels (not present).
- Generalization beyond the welding processes and materials represented.

### Who created it?

- Upstream reference dataset: compiled from a mix of public weld defect imagery (Kaggle weld defect datasets and similar) and synthetic augmentation. The upstream model does not include any single customer's proprietary data.
- Deployed-model datasets: combine upstream reference data with customer-supplied labeled images from production welding stations. Customer datasets are not bundled in this repository; they remain the customer's property.

### Who funded it?

The upstream reference work is the author's independent work, unfunded. Deployments are customer-funded.

---

## Composition

### What do the instances represent?

Each instance is a 2D RGB image of a welded metal seam, along with one or more bounding-box annotations. Each annotation has a bounding box (xyxy in image pixel coordinates, or normalized xywh in YOLO format) and one class label from:

| Class ID | Class name | Description |
|---|---|---|
| 0 | crack | Linear discontinuity in weld metal. Includes longitudinal, transverse, crater, toe cracks. |
| 1 | porosity | Gas pockets trapped during solidification. Includes surface-breaking pores and clustered porosity. |
| 2 | spatter | Metal droplets expelled during welding that adhered to the workpiece. |
| 3 | undercut | Groove melted into the base metal adjacent to the weld toe, not subsequently filled. |
| 4 | overlap | Weld metal flowing over the base metal without fusion at the weld toe. |

These classes map to a subset of ISO 5817 imperfection categories.

### How many instances?

- Upstream reference training set: approximately 6,000 images with approximately 18,000 labeled boxes.
- Per-deployment sets vary. The shipyard pilot used 4,820 customer images (plus the upstream reference). The automotive pilot used 22,000 customer ROIs.

### What data is included?

Per image: the image file (JPEG or PNG), a YOLO-format label file, and metadata (source, capture date, original resolution).

Per annotation: class id, normalized xywh.

Per deployment-specific dataset, additional metadata:
- Station id / camera id
- Weld process (SAW / GMAW / GTAW / SMAW)
- Base material grade and thickness (when available)
- Consumable lot (when available)
- Labeling session id (for traceability)

### Is any information missing?

- Operator identity: explicitly not collected.
- Subsurface-defect ground truth: not applicable (out of scope).
- Paint / coating state: not systematically recorded.
- Pre-inspection prep (wire-brushing, slag removal): sometimes recorded, not always.

### Are there sampling or selection biases?

Yes. Known biases:

1. **Class imbalance**: Crack and Overlap are under-represented in the upstream reference dataset because they are less common in practice. The training pipeline uses class-aware sampling to mitigate.
2. **Geographic / supplier bias**: the public sources skew toward European and North American weld-process conventions. Deployments in APAC (shipbuilding, Korean automotive) need customer-supplied data to bridge.
3. **Camera geometry bias**: the upstream reference data spans many cameras and angles; deployments typically need retraining against their specific camera geometry.
4. **Labeler bias**: different datasets used different labelers with different training; inter-labeler kappa is not uniformly reported for the upstream data.

### Are there errors or redundancies?

- Some public datasets contain near-duplicate images (multiple camera angles of the same weld segment). Deduplicated at the image-hash level before training.
- Label noise exists; spot-checks on samples from public datasets identify approximately 3-5% miscategorization (primarily Spatter labeled as Porosity or vice versa).

### Does the dataset contain sensitive or PII data?

- **Upstream reference**: No faces, no license plates, no operator names. Some images contain partial views of gloved hands or welding gear.
- **Deployed datasets**: Automotive body-shop images may capture VIN stickers; VINs are redacted via OCR-based masking before storage.
- **Operator identifiers**: not present in the image data; operator metadata, if attached, is separable and is not used for training.

---

## Collection process

### How was the data collected?

- Upstream reference: downloaded from public sources (Kaggle weld defect datasets and similar), filtered for the 5 target classes, re-labeled where necessary.
- Deployed datasets: captured by in-station cameras at the customer's welding cells (Basler / IDS industrial cameras; customer-specific).

### Who collected it?

- Upstream: authors of the public source datasets (credited in each source's own documentation), plus the repository author for curation and re-labeling.
- Deployed: customer-owned cameras, customer-owned pipeline.

### How was the data labeled?

- Upstream: labels inherited from source datasets where available, normalized to the 5-class schema. Re-labeling performed by the repository author using CVAT.
- Deployed: labeling done in CVAT by a mix of customer QA engineers and external labelers. 10% double-labeling for QA. Cohen's kappa tracked per labeler.

### What is the labeling schema?

YOLO format: one .txt per image, each line `class_id x_center y_center width height` normalized to [0, 1]. Schema version 1 (the current 5-class schema).

### What was the labeling timeline?

- Upstream: spread over approximately 4 months of intermittent work.
- Shipyard deployment: 5 weeks of primary labeling, ongoing human-in-the-loop labeling thereafter.
- Automotive deployment: 3 weeks of primary labeling with a larger team, ongoing thereafter.

---

## Preprocessing

### What preprocessing was applied?

- Image decode (JPEG / PNG) to RGB tensors.
- Letterbox resize to 640x640 with 114 gray padding; bounding boxes transformed identically.
- At training time: Ultralytics' default augmentation (mosaic, mixup, HSV jitter, h/v flip) per `src/config.py`.
- At evaluation time: letterbox only; no augmentation.

### Is raw data preserved?

Yes. Preprocessed tensors are produced on-the-fly from the original image files; the dataset stores raw images.

---

## Uses

### Has the dataset been used for anything else?

- Some public source datasets have been used in academic benchmarks; we do not redistribute those sources, only reference them for training.
- Customer datasets are used only for training/evaluating the customer-specific model.

### Is there a repository linking to this dataset?

- Upstream: partial public sources linked from the README (Kaggle / Roboflow listings).
- Customer datasets: not published; remain customer property per engagement terms.

### What tasks should this dataset NOT be used for?

- Training a model that operates on subsurface imagery (UT, RT); schema mismatch.
- Training a model intended as sole authority in safety-critical contexts.
- Training a model with 4 or fewer classes (throws away class signal); this repo's schema is fixed at 5.

---

## Distribution

### Will the dataset be distributed?

- Upstream reference: individual source datasets remain at their original public locations; this repo does not redistribute them.
- Customer datasets: not distributed. Customer retains ownership.

### How will it be distributed?

- Source control: not the dataset itself (large binary), but the YAML configuration and labeling schema are in this repo.
- Artifacts: customer-specific datasets stored in customer-owned object storage (S3 / Azure Blob / on-prem MinIO).

### Subject to copyright or IP?

- Upstream sources: subject to the license of each source dataset. This repo does not override those licenses.
- Customer datasets: customer property.

---

## Maintenance

### Who maintains the dataset?

- Upstream schema: Doeon Kim (repo author).
- Per-deployment datasets: customer ML/QA team.

### How can errors or issues be reported?

- Upstream: GitHub issues on this repository.
- Per-deployment: the customer's internal issue tracker.

### Will the dataset be updated?

- The schema is considered stable at v1 (5 classes). Schema changes will be versioned.
- Per-deployment datasets are updated continuously via the human-in-the-loop labeling pipeline (see [`docs/production/labeling-pipeline.md`](../docs/production/labeling-pipeline.md)).

### Will older versions be kept?

- Schema versions are tracked.
- Per-deployment datasets: customers are advised to retain full label history for auditability (ISO 9001, IATF 16949).

---

## Cross-references

- Model card: [`governance/model-card.md`](./model-card.md).
- Ethics review: [`governance/ethics-review.md`](./ethics-review.md).
- Training config: [`src/config.py`](../src/config.py), [`data/weld_defect.yaml`](../data/weld_defect.yaml).
- Labeling pipeline: [`docs/production/labeling-pipeline.md`](../docs/production/labeling-pipeline.md).
