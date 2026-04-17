# Labeling Pipeline and Human-in-the-Loop Retraining

Once the model is in production, fresh labels come from two sources:

1. The disposition workflow (CWI or rework-cell operator confirms or disagrees with each detection).
2. Periodic blind relabeling on sampled images.

This document describes how those labels flow back into training in a way that avoids common pitfalls (selection bias, drift on the training set, catastrophic forgetting).

---

## 1. Label sources and their biases

### 1.1 Disposition labels

Each production detection becomes a ticket. The operator marks it confirm/disagree. This gives us:

- **Confirmed positives**: the detection is correct.
- **Rejected positives** (operator disagrees): false positive.

What we do *not* get from disposition alone: **false negatives**. If the model missed a defect entirely, the operator sees no ticket for it and there is no disagreement signal.

Selection bias: using only disposition labels for retraining will bias toward making the model more conservative (fewer false positives) at the cost of recall, because we only feed back what the model already saw. Mitigation: combine with blind relabeling.

### 1.2 Blind relabeling

On a cadence (weekly for most deployments), sample ~200 images from the production stream without showing the model output to the labeler. The labeler annotates as if this were fresh training data. Compare against the model's predictions post-hoc.

This gives us true false-negatives (the labeler found a defect the model missed). Use blind-relabel data as the primary source of recall-corrective training signal.

### 1.3 Active learning sampling

Instead of sampling uniformly, sample images the model is uncertain about:

- Highest variance in per-object confidence.
- Detections near the confidence threshold (in the 0.35-0.55 confidence band).
- Images where two models (current prod, challenger candidate) disagree.

This concentrates labeling cost on informative examples.

---

## 2. The pipeline

```
Production stream (images + detections)
    │
    ├─────────────────────────────────────────┐
    ▼                                         │
Operator disposition (confirm/disagree)       │
    │                                         │
    ▼                                         ▼
[DB] production_dispositions         [Sampler] uniform_weekly
    │                                         │
    │                                         ▼
    │                            [Sampler] active_learning
    │                                         │
    │                                         ▼
    │                                    [DB] labeling_queue
    │                                         │
    │                                         ▼
    │                                  CVAT instance
    │                                  (QA engineer labels)
    │                                         │
    │                                         ▼
    │                            [DB] labels_fresh (blind + active)
    │                                         │
    └──────────┬─────────────────────────────┘
               │
               ▼
       Training dataset compositor
       (train/val split, class balancing,
        stratified sampling from each source)
               │
               ▼
       Retraining job (weekly / on-demand)
               │
               ▼
       Regression test: golden set
               │
               ▼
       Model registry (new version)
               │
               ▼
       Canary deploy → full deploy
```

---

## 3. Data governance

### 3.1 Labeler QA

- 10% of every labeler's work is double-labeled by a second labeler (or the lead QA engineer).
- Inter-labeler agreement (Cohen's kappa) is tracked per labeler.
- If kappa drops below 0.75 for any labeler for two consecutive weeks, retrain that labeler against a reference set.

### 3.2 Label schema versioning

The 5 classes from `src/config.py` are the stable schema. Any change to class definitions (e.g., adding `burn-through` as a 6th class) is a schema version bump that invalidates old labels unless they are reviewed for compatibility.

### 3.3 Provenance

Each label records:
- Labeler id.
- Timestamp of label.
- Tool version (CVAT version, config).
- Source image id (back-references to production or archive).
- Review status (primary, reviewed, disputed).

Provenance is auditable. For deployments in regulated plants (ISO 9001, IATF 16949 for automotive) the audit trail is a compliance requirement, not a nice-to-have.

### 3.4 PII

Weld images are generally PII-free (no faces, no identifying marks). The exception is the operator's gloved hand occasionally entering frame; this does not constitute PII in most jurisdictions but we crop anyway.

Body-shop images may contain VIN stickers (automotive). Strip VINs via OCR-based redaction before archiving. See [`governance/data-sheet.md`](../../governance/data-sheet.md).

---

## 4. Retraining cadence

| Trigger | Cadence | Scope |
|---|---|---|
| Scheduled (weekly) | Every Monday 02:00 local | Last 7 days of labels added to training set |
| Drift alert | On-demand | Targeted labels from the drift window |
| New body model (automotive) | On-demand | Full training run against new labels |
| New consumable lot (shipyard) | On-demand | Targeted labels, fine-tune only |
| Annual | Yearly | Full retraining from scratch for auditability |

Weekly is the default. Any retraining goes through the regression test against the golden set before promotion.

---

## 5. Avoiding catastrophic forgetting

The training set grows week-over-week. To avoid the model becoming too specialized to the most recent week's distribution:

- Retain the original training set as a permanent anchor.
- New labels are added with a decay weight: freshly-labeled images have weight 2.0, images older than 90 days have weight 1.0, images older than 1 year have weight 0.7 (tunable).
- Stratified sampling per class ensures the rare classes (Crack, Overlap) are upsampled at training time even as the majority classes accumulate.

---

## 6. Promotion gate

A new model version is promoted to production only after passing all:

1. **Golden set mAP**: must be >= prior version - 0.01.
2. **Per-class precision** on the last 30 days of production images: within 0.02 of prior version.
3. **Per-class recall** on the weekly blind-relabel set: within 0.02 of prior version, and > gating threshold for Crack class.
4. **Latency benchmark**: p95 within 1 ms of prior version on same hardware.
5. **Smoke test on edge device**: 100-image inference run with numerical comparison vs PyTorch original.

Gate violations are surfaced to the ML lead; the deployment does not auto-proceed.

---

## 7. Tooling

- Labeling: CVAT (self-hosted, mounted from plant-local S3 for images).
- Label storage: Postgres, JSONL exports versioned in the model registry.
- Retraining orchestration: the plant's existing orchestrator (Airflow / Prefect / cron on the ML node).
- Model registry: MLflow or similar, with the artifact store in plant-local S3.
- Regression tests: `pytest` with the golden-set images baked in as a fixture.

---

## 8. Data volume expectations

From the shipyard deployment, first 90 days:

- Production detections: 117,400.
- Operator-reviewed dispositions: 113,200.
- Confirmed positives: 98,800.
- Rejected positives (false positive): 14,400.
- Blind-relabel samples: 1,620 (18 weekly cycles x ~90 per cycle, minus a few missed cycles).
- Net new labels added to training: ~11,200 (selective sampling).
- Retrains run: 14 (13 scheduled + 1 drift-triggered).
- Retrains promoted: 9 (5 failed the gate — 4 on golden-set regression, 1 on Crack recall).

The 5 failures are important. The gate catches regressions that would otherwise reach production.

---

## 9. References

- Drift detection: [`docs/production/monitoring-drift.md`](./monitoring-drift.md).
- Shipyard retraining cycle: [`docs/case-studies/shipyard-pipeline.md#61-dataset`](../case-studies/shipyard-pipeline.md).
- Automotive label flow: [`docs/case-studies/automotive-body-shop.md#10-lessons-learned`](../case-studies/automotive-body-shop.md).
- Data sheet: [`governance/data-sheet.md`](../../governance/data-sheet.md).
