# Ethics Review: Weld Defect Vision

Deploying computer vision in a manufacturing plant has ethical implications beyond the technical ones. This document records the review conducted for this project: the questions asked, the positions taken, and the residual risks. It is not a legal document and does not substitute for a per-deployment review at the customer site.

---

## 1. Worker monitoring

### The question

Can the detection telemetry be used to rank, rate, or discipline individual welders? If the MQTT feed of "defect detected" is attributed to a welder-ID from the MES, the stream effectively becomes a per-worker performance record.

### The position taken

**No without labor engagement.** Per-operator performance analysis requires, at minimum:

1. Explicit agreement with labor representation (union, works council, or equivalent).
2. A defined use policy that describes what the data is used for (e.g. "process improvement only; not input to performance reviews") and what it is not used for.
3. Worker awareness: operators are informed the system is in place and what it measures.
4. A grievance process if a welder believes they are being unfairly evaluated.

In deployments without labor engagement, the analytics pipeline aggregates at the shift or cell level, not the individual. The MES feed carries a welder-id field but the analytics views do not render it; this is a technical guard but it is not a substitute for the policy guard.

### Residual risk

A determined plant could query the data directly and produce per-operator reports. The system architecture does not prevent this. The guard is organizational and contractual.

---

## 2. False-positive cost

### The economic cost

A false positive that escalates to line-stop has direct economic cost:

- Shipyard: ~5 min of robot downtime + CWI dispatch + senior welder pager. Estimated USD 40-120 per false line stop.
- Automotive: ~5-15 min of line-stop + downstream rework cell capacity consumed. Estimated USD 200-600 per false line stop at a high-throughput plant.

### The human cost

Welders who are repeatedly (and wrongly) told their welds are defective lose confidence in the system and in themselves. A poorly-calibrated system is, over time, demoralizing. Mitigation:

- Conservative thresholds at cutover; loosen only as confidence grows.
- Clear operator-facing UX showing the confidence level and the inspection image.
- Fast override path: operator or CWI can override the model's call with a logged comment.
- Weekly review of override rates by class; if override rate is consistently high for any class, threshold re-tune.

### The quality cost

Chronically high false-positive rates cause "alert fatigue": operators start ignoring alerts, including the true positives. This is the single most dangerous failure mode of a production ML system. Monitoring for it requires:

- Tracking operator action-latency (time from alert to acknowledge/override) as a proxy for attention.
- Mandatory weekly review of the false-positive rate; any single-class rate above 15% for two consecutive weeks is a retraining priority.

---

## 3. False-negative cost

### The safety cost

A missed crack in a load-bearing structural weld can, in principle, cause structural failure. In the shipyard case, missed welds could affect hull integrity. In the automotive case, missed welds in underbody reinforcements could affect crashworthiness.

The model is explicitly NOT the sole defense. Missed welds should be caught by:

- The CWI's random audit (in shipyard).
- Downstream UT / MT spot checks (in shipyard).
- Downstream laboratory destructive testing on a batch sample (in automotive).
- Customer-facing warranty-driven feedback (long latency, but it exists).

### The model's recall bar

Crack recall is the gating metric precisely because of this cost asymmetry. A Crack recall below the deployment-specific floor (0.80 in the shipyard case) is a reason to withhold cutover; no aggregate mAP number overrides this.

---

## 4. Data handling

### Images

Weld images are generally not PII. Exception: body-shop images with VIN stickers; VINs are redacted via OCR-based masking before storage.

### Retention

- Hot storage (NVMe on edge): 14 days.
- Warm storage (plant S3): 6 months.
- Cold storage: 2 years, compressed. Required retention for ISO 9001 / IATF 16949 audit purposes.
- Deletion on request: the customer's QA lead can request deletion of a specific image set. Deletion propagates through archives.

### Model artifacts

- Models are versioned and archived; older versions are retained for rollback up to 12 months post-retirement.

---

## 5. Automation scope

### What the system decides autonomously

- Per-ROI classification (class, confidence).
- Per-body aggregate pass/fail in the automotive case (via the rule engine).
- Per-weld ticket creation in the shipyard case.

### What the system flags but does NOT decide

- Final defect disposition. This is always a human decision (CWI or rework-cell operator).
- Line stop vs continue: in shipyard, the Andon is advisory; a welder can acknowledge and continue. In automotive, line stops are enforced by the PLC, but a line lead can override with logged rationale.

### What the system never decides

- Whether to ship a body / hull. This is outside the system's scope.
- Whether a welder is under-performing. This is an HR process, not an ML one.

---

## 6. Failure handling

### When the model is wrong

- Operator override is always available.
- Override events are logged and reviewed.
- Patterns of override trigger retraining.

### When the system is down

- Failsafe to prior inspection method (manual CWI walk, or prior template-matching system).
- Edge node has a systemd watchdog that restarts on sustained failure.
- Plant ops has a runbook for a 24h outage that routes all welds to the manual QA path.

### When the model is right but the consequence is wrong

- A true-positive that causes unnecessary escalation is logged, reviewed, and may lead to a threshold tightening.
- Escalation thresholds are joint decisions between the deployment team and the customer QA lead, not unilateral.

---

## 7. Consent

### Is operator consent required?

In most jurisdictions, monitoring of industrial machinery and its outputs (the welded part) does not require individual operator consent. However:

- Operators should be informed the system is in place.
- Deployments in jurisdictions with stronger worker-monitoring laws (Germany, France, parts of the Nordics, Korea depending on the plant's labor agreement) may require consultation with works councils or labor representation.
- The default position is to engage labor early, before deployment, and to document the engagement.

### Is customer consent required for data sharing?

The upstream repo does not include customer data. Customer datasets remain customer property and are not shared. Any use of customer images for case studies or external demonstration requires explicit written permission.

---

## 8. Regulatory envelope

- **ISO 9001** (quality management): applies to the customer's overall QA process; the model is an input, not a standalone compliance tool.
- **IATF 16949** (automotive): stricter per-part traceability; requires that inspection decisions be auditable and repeatable. Deterministic inference + versioned models + full label/decision history meets this bar.
- **ABS / KR / DNV** (shipbuilding class rules): the model does not alter the class requirements. Certified inspection remains required; the model raises coverage.
- **Export control**: YOLOv8 is permissively licensed; no export-control issues in the default configuration. Deployments to entities on sanctions lists require case-by-case review.

---

## 9. Bias considerations

- **Class imbalance bias**: addressed in training via class-aware sampling and loss weighting; see [`benchmarks/results/class-balance-impact.json`](../benchmarks/results/class-balance-impact.json).
- **Camera geometry bias**: each deployment retrains against its own camera geometry; the upstream model is not deployed directly in production.
- **Process bias**: different welding processes (SAW vs GMAW) produce different bead appearances; deployments must include their specific processes in training data.
- **Labeler bias**: inter-labeler kappa is tracked; labelers below threshold are retrained.

---

## 10. Residual risks and acknowledgment

The following risks are acknowledged and not fully mitigated:

1. **Dual-use**: the telemetry could be used for worker surveillance. Mitigation is organizational, not technical.
2. **Model decay over time**: mitigated by monitoring + retraining, but a sudden undetected drift could produce a window of incorrect decisions.
3. **Adversarial robustness**: the model has not been tested against adversarial inputs. In the industrial setting, adversarial inputs are not a typical threat model, but we note it.
4. **Supply-chain trust**: Ultralytics, PyTorch, TensorRT are all third-party dependencies. We monitor advisories but cannot guarantee against a supply-chain compromise.

---

## 11. Review cadence

- The ethics review is updated when:
  - A new deployment type or use case is added.
  - A material change is made to the model or the decision thresholds.
  - Regulatory environment changes (new labor law, new industry standard).
- A scheduled review is conducted annually at minimum.

---

## 12. Sign-offs (deployment-specific)

Each deployment records the following sign-offs at cutover:

- Customer QA lead (ownership of thresholds).
- Customer welding engineer (process implications).
- Labor representation or HR (operator monitoring implications).
- ML owner (model readiness).

Upstream (this repo) does not carry customer-specific sign-offs; that lives in per-deployment documentation.

---

## Cross-references

- Model card: [`governance/model-card.md`](./model-card.md).
- Datasheet: [`governance/data-sheet.md`](./data-sheet.md).
- Shipyard deployment (CWI buy-in and FMEA update): [`docs/case-studies/shipyard-pipeline.md`](../docs/case-studies/shipyard-pipeline.md).
- Automotive deployment (Andon override policy): [`docs/case-studies/automotive-body-shop.md`](../docs/case-studies/automotive-body-shop.md).
- Drift monitoring (alert fatigue mitigation): [`docs/production/monitoring-drift.md`](../docs/production/monitoring-drift.md).
