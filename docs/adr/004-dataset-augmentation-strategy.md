# ADR 004: Dataset Augmentation Strategy

- **Status**: Accepted
- **Date**: 2026-04-17

## Context

Industrial defect detection datasets are small relative to general-object-detection datasets. Real weld-defect imagery from any single production line is constrained by:

- Defect rate: most welds are good (imbalance 10:1 or worse).
- Labeling cost: each defect requires a trained welder to annotate.
- Confidentiality: production customers are unlikely to share their imagery publicly.

Options for training data augmentation:

1. **No augmentation**: train on raw images only.
2. **Traditional augmentation**: flip, rotate, color jitter, crop.
3. **Mosaic + MixUp** (YOLOv8's built-in aggressive augmentations).
4. **Synthetic data generation**: GAN or diffusion-based defect image synthesis.
5. **Physics-based simulation**: render defects via 3D scene modeling.
6. **Active learning with human-in-the-loop**: prioritize annotation on edge cases.

## Decision

Adopt **Option 3 (Mosaic + MixUp + HSV jitter + flip) as default** via YOLOv8's built-in augmentation pipeline. Combine with **Option 6 (active learning)** for the production feedback loop, documented separately in `docs/production/labeling-pipeline.md`.

Explicitly defer Options 4 and 5 until we have production data suggesting the default augmentation ceiling is actually reached.

## Consequences

### Positive

- **Fast path to a production-capable model**: YOLOv8's default augmentations are battle-tested on COCO and similar scales; they transfer well to defect detection.
- **Strong data efficiency**: Mosaic creates 4-image composites, effectively quadrupling the diversity of the training signal per epoch.
- **MixUp cross-class regularization**: reduces the class-imbalance overfitting that plagues raw training on 10:1 imbalanced datasets.
- **HSV jitter** handles the lighting variability between different factory shop floors without requiring per-site retraining.
- **Horizontal flip is safe** for weld imagery (weld defects are approximately orientation-invariant).
- **Vertical flip is disabled by default** (welds have gravity — a crack pattern upside-down looks unnatural and would degrade training).

### Negative

- **Hidden from users**: augmentation happens inside the training loop; model-card consumers don't see it unless they read this ADR.
- **Not tuned for edge-case defects**: the Crack class in particular benefits from additional targeted augmentation (rotation 15-30 degrees, brightness variation). The default pipeline misses this.
- **Risk of over-augmentation for small datasets**: Mosaic on a 200-image dataset can amplify label noise. For small datasets, disable Mosaic in `src/config.py`.
- **No synthetic data generation**: we can't produce the extreme rare defects (e.g., massive overlap failures) that are too rare to collect naturally.

### Mitigations

- **Per-class augmentation tuning**: `src/config.py` exposes `AUGMENTATION_OVERRIDES` allowing per-class toggles. Teams adapting this for a specific factory can disable Mosaic for very small datasets or increase rotation for Crack.
- **Augmentation ablation baselines**: `benchmarks/accuracy_benchmark.py` includes an `--augmentation-ablation` flag that runs training with each augmentation toggled to quantify contribution.
- **Synthetic data as a future lever**: `docs/adr/` can be amended with an ADR-008 (TBD) once we hit a plateau on a specific class and need synthetic generation.

## Alternatives considered

### Option 1 — No augmentation

Rejected. On the typical 500-1500 image industrial dataset, no augmentation leads to rapid overfitting and poor field performance.

### Option 2 — Traditional augmentation only

Rejected. Flip + rotate + color jitter leaves Mosaic + MixUp gains on the table. These two modern augmentations consistently improve small-object and small-class performance by 3-8 mAP points in the ultralytics/yolov8 release notes and third-party benchmarks.

### Option 4 — Synthetic data generation (GAN / diffusion)

Deferred. High cost: building a defect-conditional image generator is a multi-week project in itself. We would only invest if Mosaic-augmented performance plateaued on a specific customer's dataset. As of this decision we haven't hit that plateau.

### Option 5 — Physics-based simulation

Deferred. Similar reasoning to Option 4. A physics simulator for weld defects (molten metal surface dynamics, porosity formation) is a research project. Not on the critical path.

### Option 6 — Active learning in isolation

Partially accepted. Active learning is complementary, not substitute. We do both: aggressive augmentation during training, active selection during labeling. The labeling-pipeline doc covers the active learning side.

## How this constrains production deployments

- **Customer-specific tuning**: a customer with 200 images per class should disable Mosaic (`mosaic=0.0` in `src/config.py`). A customer with 1500+ images per class should keep the defaults.
- **Class-imbalance monitoring**: track per-class sample count in production retraining runs. When the minority class drops below 50 images per epoch, pause training and expand labeling before retraining.
- **Camera calibration drift**: HSV jitter handles reasonable lighting variation. If a customer changes cameras or lighting significantly, retrain — do not assume augmentation compensates.
- **Rotation limits**: default rotation is +/- 10 degrees. For welders working on non-axis-aligned pipes, increase to +/- 30 degrees in config.

## Measurement

Augmentation impact is measured via:

- **Per-class mAP@50** on the held-out test set, per augmentation setting. Default config target: 0.85+ on Crack, Porosity, Spatter; 0.75+ on Undercut, Overlap (harder classes).
- **Training loss stability**: Mosaic-enabled runs should show lower variance in the last 20 epochs vs disabled runs. If you see training instability, Mosaic is too aggressive for your dataset scale.
- **Field false-negative rate**: the real measure. Track in production; target under 3% on safety-critical defects.

## References

- YOLOv8 augmentation docs: https://docs.ultralytics.com/usage/cfg/#augmentation
- YOLOv5 augmentation study: https://arxiv.org/abs/2004.10934 (the original Mosaic paper in the YOLO line)
- `src/config.py` — augmentation config
- `src/dataset.py` — dataset pipeline calling the augmentations
- `docs/production/labeling-pipeline.md` — active learning companion
