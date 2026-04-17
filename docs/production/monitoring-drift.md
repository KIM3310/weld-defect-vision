# Monitoring and Model Drift

The model is not finished when it is deployed. This document covers the monitoring pattern used in the shipyard and automotive deployments to detect and respond to drift.

---

## 1. What we are monitoring

Four classes of signal:

1. **Infrastructure**: is the service up, latency in spec, GPU utilization.
2. **Input distribution**: does the image distribution reaching the model look like the training distribution.
3. **Output distribution**: do the model's raw detections (class mix, confidence histogram, count per image) look like the training-time reference.
4. **Ground-truth-validated**: is the model still correct, measured against CWI or rework-cell labels.

Each has a different response latency. Infrastructure alerts in seconds. Input/output distribution alerts in hours. Ground-truth validation alerts in days to weeks.

---

## 2. Infrastructure signals

| Metric | Source | Alert |
|---|---|---|
| Triton `/metrics` latency histogram p99 | Triton metrics | > 25 ms for 5 min |
| Triton queue depth | Triton metrics | > 8 for 2 min |
| Capture daemon heartbeat | MQTT | missing > 60 s |
| GPU utilization | DCGM | not in 20-85% range for 10 min |
| Orin thermal tj | tegrastats | > 78 C for 2 min |
| Edge node disk | SMART / node exporter | > 85% full |
| OPC-UA subscription health | asyncua internal state | disconnected > 30 s |

Alerts route to plant ops via PagerDuty/Opsgenie equivalents. First-line response is the plant IT on-call; ML team is second-line for model-specific incidents.

---

## 3. Input distribution drift

Even if the model is still correct in the narrow statistical sense, a shift in the input distribution is usually the earliest signal that something is changing. Detection methods in use:

### 3.1 Mean pixel statistics

For each captured image, compute mean R, G, B; mean luminance; contrast (std of luminance). Publish to the time-series database. Run a 24-hour rolling window against a 30-day reference baseline.

Alert when:
- Mean luminance shifts by > 1.5 sigma.
- Contrast shifts by > 1.5 sigma.

### 3.2 Per-image feature distribution

Run a lightweight feature extractor (e.g., the penultimate layer of an ImageNet-pretrained ResNet18, or the YOLO backbone's global-average-pooled features) on a 1% sampled subset of production images. Compute a population-level distance (KL divergence or MMD) against a reference distribution computed from the training set.

Alert on drift score above threshold. This is a lagging signal (hours to day) but catches subtler shifts than pixel statistics.

### 3.3 Ambient and process features

When available, attach the ambient light sensor reading and the weld-parameter telemetry (voltage, current, wire feed speed) to each image. Monitor these for shifts. A shift in weld voltage correlated with a defect-rate shift is a process issue, not a model issue — but the defect-detection monitoring pipeline catches it first.

---

## 4. Output distribution drift

The model's own outputs are the second-earliest signal. Track:

### 4.1 Per-class detection rate

Count of detections of each class per shift (8 hours). Compute a 30-shift rolling baseline and compare the current shift to it.

Example: over the shipyard's first 90 days, the Porosity per-shift detection count held steady at 142 +/- 31 per shift (mean, std). In week 9 post-cutover it rose to 214 for one shift, 198 the next, 186 the next — a sustained shift. The drift alert fired at shift 2 of the run.

Alert: per-class count > mean + 2.5 sigma for 2 consecutive shifts.

### 4.2 Confidence histogram

For each class, bucket confidence scores (e.g., 20 bins between 0.25 and 1.0). Compute the per-bin count and compare against the reference histogram via chi-squared distance.

Alert: chi-squared distance above threshold for 2 consecutive windows.

### 4.3 Detection-count per image

Distribution of (detections-per-image) across the shift. A shift to more detections per image often precedes a false-positive regime change.

Alert: median detections-per-image > reference median * 1.3 for 2 shifts.

---

## 5. Ground-truth validated metrics

The gold standard. Requires labeled ground truth, which comes from:

1. **CWI / rework-cell disposition** (every ticket confirmed or disagreed): this provides a continuous confusion-matrix-like signal, but with selection bias (you only get labels for what the model flagged).
2. **Periodic blind holdout evaluation**: every N days, randomly sample 200 weld images, have a CWI label them without seeing the model output, then score the model. This catches false negatives.
3. **Golden set regression**: a fixed set of 500 images labeled once, run through every new model version before promotion. Catches model degradation across retraining.

### 5.1 The confirmation-based confusion matrix

From the ticket workflow: every detection has a disposition (confirm / false-positive). We can compute:

- **Per-class precision**: fraction of detections confirmed.

We cannot compute per-class recall from confirmations alone (we do not see what the model missed). Recall requires the periodic blind holdout.

### 5.2 Cadence

| Check | Cadence | Resource cost |
|---|---|---|
| Infrastructure | continuous | negligible |
| Input distribution | hourly aggregates, daily review | 1% sampled feature extraction |
| Output distribution | continuous aggregates, daily review | negligible |
| CWI-confirmation confusion matrix | daily aggregate | the CWI flow anyway |
| Blind-holdout evaluation | weekly | ~4 hours CWI labeling + 1 hour eval |
| Golden-set regression | per model version | ~1 hour eval |

---

## 6. The response playbook

When a drift alert fires, the response is:

1. **Classify the alert**: infra, input-dist, output-dist, or ground-truth.
2. **Confirm the signal**: is it a single noisy window, or sustained?
3. **Identify the cause**: consumable lot change, camera fault, seasonal ambient shift, genuine defect-rate increase, model degradation.
4. **Respond**:
   - If process: notify welding engineer.
   - If camera: swap or recalibrate.
   - If model: evaluate against golden set, decide on retrain vs rollback.
5. **Close the loop**: document in the ops log; update thresholds if appropriate.

### 6.1 Example: the wire-lot drift event

From the shipyard case study, week 9 post-cutover:

- **Day 1**: Porosity per-shift detection count jumped from 142 to 214. Output-dist alert.
- **Day 2**: CWI-confirmation precision dropped from 0.84 to 0.71 on Porosity. Ground-truth alert.
- **Day 3**: Triage: compared input-distribution statistics, found no change in luminance or contrast.
- **Day 3**: Pulled process telemetry, found the consumable lot number had changed on Day 1. Cross-checked with the consumable store: new lot.
- **Day 4**: Collected 80 new Porosity images from the new lot; labeled. Added to training set.
- **Day 6**: Retrained model; regression-tested against golden set (no degradation) and against the new lot (recovered precision to 0.81).
- **Day 7**: Deployed new version. Monitored for 48h; stable. Retired old version.

Total: 7 days from alert to recovered performance. This is the intended operational tempo.

---

## 7. Dashboards

Plant ops dashboards show:

- **Health panel**: service up, queue depth, latency p99, thermal, disk.
- **Volume panel**: detections per shift by class, grouped by station.
- **Drift panel**: input-distribution drift score, output-distribution chi-squared, last golden-set mAP.
- **Accuracy panel**: per-class precision from confirmation, false-positive rate, last blind-holdout recall.

---

## 8. Alert hygiene

Drift monitoring produces false alerts. Mitigations:

- Require 2 consecutive windows for all drift alerts (eliminates single-shift noise).
- Monday morning and post-weekend alerts are suppressed for the first hour of the first-shift (gives the line time to warm up).
- Changes to consumable lots, WPS, or body model are logged and the alert system suppresses drift alerts for the first 24 hours post-change (to avoid re-alerting on expected shifts).

---

## 9. References

- Labeling pipeline (how drift events feed retraining): [`docs/production/labeling-pipeline.md`](./labeling-pipeline.md).
- Monitoring signals and retraining cadence described in the shipyard case: [`docs/case-studies/shipyard-pipeline.md#66-false-positive-cost`](../case-studies/shipyard-pipeline.md).
