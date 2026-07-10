# Case Study: Automotive Body Shop Spot-and-MIG Weld QA

Deployment of `weld-defect-vision` in an automotive body-in-white (BIW) line at a Tier-1 OEM's domestic plant. This is a higher-throughput, higher-cadence deployment than the shipyard case: one vehicle passes through the weld inspection station every 42 seconds, with the target being a defect classification decision every ~2 seconds per weld region of interest.

Customer is anonymized as **"Plant-C"**. Volumes, takt times, and numbers are representative of a real engagement shape and not a literal audit of any single customer.

---

## 1. Plant context

Plant-C produces midsize sedans. The body shop runs at takt time 42 seconds per body. BIW welding is a mix of:

- **Resistance spot welding** (RSW): ~4,800 spots per body, done by robots with servo guns.
- **MIG/MAG welding**: short continuous seams at roof-to-side-body joints, underbody reinforcements, rear trunk pan, etc. ~38 m of continuous seam per body.
- **Laser welding**: roof ditch only, outside the scope of this deployment.

The deployment target is a **line-of-sight inspection station** after the framing robots and before the underbody seal line. Inspection is 100% automated and the station has 14 seconds of dwell time per body. Within those 14 seconds the system must:

1. Acquire images from 6 cameras distributed around the station.
2. Classify weld ROIs (regions of interest: specific spot-weld locations and specific MIG seam segments).
3. Issue a pass/fail per weld.
4. Aggregate per-body pass/fail.
5. Trigger Andon on fail.

Practical target: decision per ROI in ~2 seconds. The repo's existing 5-class model is used for the MIG seam defects. Spot-weld defect classification uses a separate trained head (not covered in this doc; it is a regression model on the indentation geometry).

### 1.1 Problem framing

The existing inspection pipeline is **teach-in template matching**: the CCD cameras capture images, the image processor (an off-the-shelf industrial vision controller) runs geometric feature checks (diameter, color, texture) against a teach-in template. The system catches:

- Spot welds that fail the indentation diameter threshold (reasonable accuracy).
- Missing welds (good accuracy).

The system does *not* catch:

- MIG seam surface defects (porosity clusters, undercut) on the side-body-to-roof joint.
- Subtle spatter on a customer-visible Class A surface.
- Overlap defects where a seam has doubled-back onto itself.

Plant-C's cost of a missed defect: a single class-A spatter defect that reaches paint shop and back translates to ~USD 80 rework or ~USD 350 scrap if the panel is unrepairable. A missed underbody porosity that causes water leakage in the customer's garage is worth thousands in warranty cost.

### 1.2 The scope negotiated

**Phase 1**: deploy vision ML on MIG seam defect detection at the body shop inspection station. Use `weld-defect-vision` as the core model (with retraining on customer-supplied MIG imagery). Output integrates with the existing Andon infrastructure. This case study covers Phase 1.

**Phase 2 (not covered here)**: extend to spot-weld defects with a separate model.

---

## 2. Throughput and latency budget

42-second takt x 1 body per cycle. 14-second dwell at the inspection station. 6 cameras, each with an average of 8 ROIs to classify per body. Total ROIs per body: ~48. Target per-ROI decision time: ~250 ms sustained (leaving headroom for the aggregation stage and the Andon handshake).

Per-image inference latency must be at or below 200 ms on the selected hardware.

### 2.1 Hardware selection

Unlike the shipyard case (one Orin per station), Plant-C wanted centralized inference with camera feeds aggregated. The deployed architecture:

- **Per-station compute**: 1U server with 2x NVIDIA L4 GPUs.
- **Inference stack**: Triton Inference Server with the YOLOv8 ONNX model compiled to TensorRT FP16. INT8 was evaluated but mAP regression on the Overlap class (0.68 → 0.59) was judged too high for a customer-visible surface; FP16 was the tradeoff that kept latency acceptable and accuracy in spec.
- **Batching**: dynamic batching enabled with max batch size 16 and 10 ms queue delay. Since 6 cameras fire roughly synchronously when a body enters the station, the batch fills within the queue delay.
- **Network**: 10 GbE from cameras (GigE Vision) to station server; 1 GbE to plant MES.

Measured inference latency:

| Batch size | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| 1 | 9.2 | 12.1 | 15.0 |
| 4 | 14.0 | 17.3 | 21.1 |
| 8 | 22.8 | 26.2 | 31.5 |
| 16 | 38.7 | 44.1 | 51.8 |

At batch 8 (the realistic operating batch during simultaneous camera trigger) the P95 is 26.2 ms. Even with pre-processing and Triton overhead, per-ROI decision is sub-100 ms, well within the budget.

---

## 3. Architecture

```
Camera (x6) ───GigE Vision───┐
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Image Capture Daemon │   (python, asyncio, harvesters)
                   │ ROI extraction       │
                   └──────────┬───────────┘
                              │ gRPC (Triton client)
                              ▼
                   ┌──────────────────────┐
                   │ Triton Inference     │
                   │ Server (2x L4 GPUs)  │
                   │ YOLOv8 ONNX + FP16   │
                   └──────────┬───────────┘
                              │ detections
                              ▼
                   ┌──────────────────────┐
                   │ Aggregator           │   (pass/fail decision per body)
                   │ Rule engine          │
                   └──────────┬───────────┘
                              │ pass/fail + per-ROI details
                              ▼
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────────┐
    │ Andon (PLC via   │           │ MES (Kafka topic)    │
    │ OPC-UA)          │           │ plant_c.biw.defects  │
    └──────────────────┘           └──────────────────────┘
                                               │
                                               ▼
                                   ┌──────────────────────┐
                                   │ Nexus-Hive analytics │
                                   │ (warranty model      │
                                   │  correlation)        │
                                   └──────────────────────┘
```

### 3.1 ROI extraction

The capture daemon knows the geometry of each body (from the MES body-in-station signal). It knows that for a given body model, Camera 3's frame contains ROIs for roof ditch spot-welds 12-18 and the side-body-to-roof MIG seam left 400 mm. The daemon crops to 640x640 ROIs and passes them in a batch to Triton.

The mapping from body model + station position + camera to ROI list is captured in a YAML config that is versioned with the body model CAD release:

```yaml
body_models:
  sedan_Z24_ph2:
    station_WQ_inspection:
      camera_3:
        rois:
          - id: roof_ditch_weld_12
            crop: [212, 148, 852, 788]
            class_policy: spot_indent
          - id: roof_side_mig_seam_left_400mm
            crop: [80, 820, 720, 1460]
            class_policy: mig_defect
```

The `class_policy` controls which model is used: `mig_defect` → the `weld-defect-vision` YOLOv8 model; `spot_indent` → a separate model not covered here.

### 3.2 Aggregator rules

The per-body pass/fail is a rule over the per-ROI detections:

- Any Crack detection with conf >= 0.55: **fail body** (Andon red).
- Any cluster of 4+ Porosity detections on a single MIG seam: **fail body** (Andon red).
- Any Undercut detection with bbox area above threshold: **fail body** (Andon red).
- Spatter on Class A surface (roof ditch, visible rear pillar): **rework ticket** (Andon orange), body continues.
- Spatter on non-Class A surface: **log only**.
- Overlap on any seam: **rework ticket** (Andon orange).

Thresholds are set in the same YAML that defines the ROIs. A change-control process requires both Plant-C's QA engineer and the body shop lead engineer to sign off on any threshold change.

---

## 4. Integration with Andon

The BIW Andon is a three-tier system:

- **Green**: body passes, releases to next station.
- **Orange**: body has a rework ticket but continues down the line; rework happens at the downstream rework cell.
- **Red**: body is stopped at the inspection station and the line lead is summoned.

Integration is via the plant's existing Andon PLC over OPC-UA. The aggregator writes to three specific tags:

- `Andon.Inspection.Pass` (BOOL)
- `Andon.Inspection.Rework` (BOOL)
- `Andon.Inspection.Stop` (BOOL)

The PLC takes those values and drives the physical tower lights and the line-stop interlock. The inspection station is the only node allowed to assert `Stop`; other stations can request rework. Latency from Triton detection to PLC tag write is measured at 38 ms median, 55 ms p95.

---

## 5. Model serving

Triton config (`serving/triton/model_repository/weld_defect/config.pbtxt`):

- Input: `images` tensor, shape `[-1, 3, 640, 640]`, FP16.
- Output: `detections` tensor, shape `[-1, 25200, 85]` (YOLOv8's default).
- Dynamic batching: max batch size 16, preferred batch sizes [4, 8, 16], max queue delay 10 ms.
- Instance group: 2 instances per GPU, 2 GPUs → 4 concurrent inference streams.

Triton deployment is containerized (`nvcr.io/nvidia/tritonserver:25.01-py3`). Model repository is mounted as a read-only volume from a versioned S3 bucket (`s3://plant-c-models/weld_defect/<version>/`).

Model versioning: `weld_defect` has subdirectories `1/`, `2/`, etc. Rolling deployment: load a new version, shift traffic, keep old version warm for 48h for rollback.

See [`serving/README.md`](../../serving/README.md) for the full configuration walkthrough.

---

## 6. Rollout

### Week 1-3: data collection

Captured 22,000 labeled ROIs across 45 shifts. Labeling done in CVAT by a mix of Plant-C's QA engineers and an external labeling vendor. QC step: 15% double-labeled by two independent labelers; inter-labeler agreement measured at 0.82 Cohen's kappa, which was acceptable.

### Week 4-6: model training and accuracy validation

Fine-tuned YOLOv8s from `yolov8s.pt` against the Plant-C labels. mAP@50 at end of training (on Plant-C holdout):

| Class | mAP@50 | Precision | Recall |
|---|---|---|---|
| Crack | 0.84 | 0.80 | 0.76 |
| Porosity | 0.91 | 0.88 | 0.89 |
| Spatter | 0.94 | 0.92 | 0.93 |
| Undercut | 0.80 | 0.78 | 0.74 |
| Overlap | 0.72 | 0.75 | 0.68 |

Accuracy acceptance gate was set by Plant-C's QA: aggregate precision >= 0.85 on Class-A-affecting defects, aggregate recall >= 0.80 on safety-affecting defects (Crack, Undercut). Both gates passed.

### Week 7-8: station integration and commissioning

- Installed 6 cameras with custom brackets designed by a Plant-C fixture engineer. All camera-to-body distances calibrated against a reference fixture.
- Ran the station in shadow mode while the existing teach-in system remained authoritative.
- Tuned: 5 ROIs were remapped after finding their crops were drifting with body-position variance; fixed by using the body's reference-pin position (read from MES) to re-center crops on-the-fly.

### Week 9: shadow mode measurement

Shadow mode ran for one week, with all detections logged and periodically audited against the existing teach-in system and against rework-cell ground truth. 1 week = ~24,000 bodies = ~1.15M ROIs evaluated. Shadow mode results:

- ML system agreement with teach-in on pass/fail: 94.1%.
- On disagreements (5.9%): rework cell followup showed ML was correct 71% of the time, teach-in was correct 19% of the time, neither caught the real issue 10% of the time.
- False positive rate (bodies routed to rework by ML that were actually good): 1.8%.
- False negative rate (defective bodies that ML passed but rework-cell found): 0.4%.

Plant-C QA accepted these numbers and approved cutover.

### Week 10: cutover (live)

Cutover on a Monday during first shift. Line lead briefed, downtime planned for 12 minutes of commissioning checks. Live from body #218 of the shift.

First day live: 6,400 bodies, 118 rework tickets, 4 line stops (3 confirmed critical defects, 1 false positive — a porosity cluster that was actually a water spot from a prior station; this image was queued for retraining).

### Week 11-14: stabilization

Added the water-spot class to the ignored-classes list for two specific ROIs. Logged detections that were marked "not a defect by rework cell" and routed them to the labeling queue. Retrained at end of week 14 with 1,100 additional labels.

---

## 7. Results (first 90 days)

### 7.1 Headline numbers

| Metric | Before | After | Delta |
|---|---|---|---|
| Escapes (defects reaching paint) per 1,000 bodies | 8.4 | 2.1 | -75% |
| Warranty-correlated weld defects per 10,000 bodies | 3.2 | 0.9 | -72% |
| Bodies sent to rework at BIW exit | 4.1% | 2.8% | -32% |
| Repeat rework (bodies re-reworked) | 0.6% | 0.2% | -67% |
| Mean rework-resolution time | 18 min | 9 min | -50% |
| Takt-time hit (line stops caused by inspection) | 1.1% | 0.8% | -27% |
| Inspection station dwell time used | 10.2s / 14.0s | 11.8s / 14.0s | +1.6s |

The "-67% repeat rework" figure is the one Plant-C QA cares about most: it means the defects caught and routed to rework are the right defects, rather than escaping rework and returning.

### 7.2 Per-class production performance

On first 30 days live:

| Class | Recall | Precision |
|---|---|---|
| Crack | 0.91 | 0.82 |
| Porosity | 0.89 | 0.85 |
| Spatter | 0.95 | 0.87 |
| Undercut | 0.84 | 0.80 |
| Overlap | 0.73 | 0.76 |

### 7.3 Uptime

Station availability over first 90 days: 99.3%. Downtime breakdown:

- Scheduled maintenance: 0.4%.
- Camera fault (failed pixel run on Camera 4, swapped): 0.2%.
- Triton server reboot (memory leak in a custom preprocessor; fixed in update): 0.1%.

---

## 8. Integration with MES and warranty data

The aggregator publishes every per-body result to a Kafka topic `plant_c.biw.defects`:

```json
{
  "body_id": "Z24-2026-WK16-0441",
  "timestamp": "2026-04-16T08:42:17.441Z",
  "body_model": "sedan_Z24_ph2",
  "decision": "rework",
  "rois": [
    {
      "id": "roof_side_mig_seam_left_400mm",
      "camera": "3",
      "detections": [
        {"class": "porosity", "conf": 0.84, "bbox": [...]},
        {"class": "porosity", "conf": 0.79, "bbox": [...]}
      ]
    }
  ]
}
```

Downstream consumers:

1. **MES rework cell** — creates rework ticket.
2. **Nexus-Hive data warehouse** — stores detection events for later correlation with warranty claims. Six-month rolling correlation between "this body had an undercut detected at BIW" and "this body had a water-leak warranty claim" informs threshold tuning.
3. **Process control dashboard** — real-time per-shift defect rates by class and by welding-cell, consumed by the production engineer.
4. **Retraining pipeline** — detections marked as disagreements by the rework cell flow to the labeling queue. See [`docs/production/labeling-pipeline.md`](../production/labeling-pipeline.md).

---

## 9. Operator UX

Unlike the shipyard case, the inspection station is almost entirely automated. Operator interaction happens at three points:

### 9.1 Line-stop handling

When the system asserts `Stop`, the line lead is paged. The line lead's tablet shows:

- The offending body id and model.
- The image with the bbox overlay.
- The class and confidence.
- Two options: `Confirm — route to red-tag cell` (body out of line for detailed rework), or `Override — pass body` (with a mandatory comment).

Override is logged and monitored. An override rate above 8% for any single week triggers a review of thresholds (no individual override rate was above 3% in the first 90 days).

### 9.2 Rework cell tablet

Rework cell operators see the rework queue with per-body detection details. They confirm or disagree with each detection after reworking. Their confirm/disagree signal is the primary feedback into the retraining labeling queue.

### 9.3 Daily shift review

The production engineer reviews the per-shift dashboard each morning. They look for trends such as a rising class-specific defect rate, a specific welder-cell with an anomalous rate, or class-distribution drift.

---

## 10. Lessons learned

### 10.1 Batching and latency

Dynamic batching with a 10 ms queue delay was critical. Without it, single-image Triton invocations at 9 ms each would have burned GPU utilization and driven the L4 budget higher. With batching, average GPU utilization is 38% — headroom to add Phase 2 (spot-weld model) without new hardware.

### 10.2 ROI geometry

Body-position variance at the inspection station has a standard deviation of ~2 mm in X and ~3 mm in Y. Without the body-reference-pin correction step in the capture daemon, this would have caused ROI drift and a persistent low-grade false-positive rate. The pin-correction step is the highest-value 50 lines of code in the entire deployment.

### 10.3 FP16 vs INT8

On the customer-visible Class A surfaces, the 9-point mAP regression on Overlap under INT8 was unacceptable. FP16 was the right tradeoff: latency impact was +3 ms at batch 8 (25 ms vs 22 ms), well within budget. On internal panels we considered a per-class precision mode (FP16 for Class-A ROIs, INT8 for others) but rejected as operational complexity for marginal savings.

### 10.4 Model versioning discipline

The model is versioned against the body model CAD release. Any new body model (new generation, facelift) requires a retraining cycle with new labels. Plant-C's PLM has the body model as a first-class entity; the model registry is linked to the PLM so the retraining trigger is automatic.

### 10.5 Labeling cost

Labeling dominated the project cost: 60% of total budget. Plant-C's long-term mitigation is the human-in-the-loop labeling pipeline, which after 6 months of operation produces roughly 2,000 cleanly-labeled new ROIs per week at marginal additional cost (they fall out of the rework cell confirmation workflow anyway).

### 10.6 Andon integration

The OPC-UA tag-write pattern was the least invasive integration approach. We considered a direct MES API write (REST) but OPC-UA to the existing Andon PLC was a 10-minute PLC tag add vs a 3-week MES API change. Choose the integration path that touches the fewest proprietary systems.

### 10.7 Shadow mode is non-negotiable

The 1-week shadow mode caught the water-spot-as-porosity issue that would have caused a first-day outage if we had cutover directly. Shadow mode cost was one week of delayed value; avoided cost was immeasurable.

### 10.8 Gauge R&R

Plant-C required a formal Gauge R&R study before cutover. The model was treated as one "appraiser" alongside three human appraisers on a 25-body repeat-measurement plan. Repeatability (same-appraiser repeat measurement agreement) for the model was 0.99 (model is deterministic given the same input); reproducibility (appraiser-to-appraiser) against the human pool was 0.86. Both in spec.

---

## 11. What would be next

- **Phase 2**: spot-weld defect regression model trained on indentation geometry.
- **Phase 3**: extend to laser-weld roof ditch QA (new camera and optics, new model).
- **Long tail**: integrate the upstream weld-parameter telemetry (voltage, current, wire feed speed per weld) with the downstream defect detection. Process-informed detection model that uses both image and weld-parameter features. 12-month horizon.

---

## 12. Cross-references

- [`docs/production/model-serving.md`](../production/model-serving.md) — Triton vs alternatives.
- [`serving/triton/model_repository/weld_defect/config.pbtxt`](../../serving/triton/model_repository/weld_defect/config.pbtxt) — the actual Triton config used.
- [`integrations/opc-ua/client.py`](../../integrations/opc-ua/client.py) — OPC-UA pattern used for Andon PLC.
- [`integrations/kafka/producer.py`](../../integrations/kafka/producer.py) — Kafka pattern for warranty correlation feed.
- [`governance/model-card.md`](../../governance/model-card.md) — model card for the deployed YOLOv8 variant.

## 13. Related projects

- **[AegisOps](https://github.com/KIM3310/AegisOps)** — the operator handoff pattern used for line-stop escalations.
- **[Nexus-Hive](https://github.com/KIM3310/Nexus-Hive)** — the warranty-correlation analytics layer consuming the Kafka defect feed.
