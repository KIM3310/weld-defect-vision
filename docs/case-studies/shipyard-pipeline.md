# Illustrative Scenario: Mid-Size Shipyard Hull Weld Inspection

> **Fictional scenario — not customer or benchmark evidence.** “Yard-K” is invented; no anonymized customer, engagement, deployment, customer-supplied dataset, hardware run, or plant outcome described here exists. Every quantity, metric, cost, timeline, quote, and before/after value in this document is fabricated and **was not measured**. Terms such as “pilot,” “production,” “observed,” and “results” describe only the hypothetical narrative. Do not cite this scenario as evidence of model, hardware, integration, safety, or business performance.

This architecture exercise explores how `weld-defect-vision` might be scoped for a robotic welding station at a hypothetical commercial shipbuilder. It is useful for requirements review and failure-mode discussion only.

---

## 1. Customer context

Yard-K operates three building docks and delivers roughly 18-22 hulls per year. The bottleneck in the QA pipeline is the hull block assembly stage, specifically the fillet and butt welds joining stiffener webs to plate. A single block is 12 m x 4 m x 3 m and contains approximately 240 linear meters of weld, distributed across approximately 1,800 individual weld segments. Welding is a mix of SAW (submerged arc, for the long straight runs) and GMAW (gas metal arc, for positional and stiffener work). Robotic welding handles around 62% of the linear meterage; the remainder is manual.

Historically, inspection is performed by two certified CWI (Certified Welding Inspectors) per shift who walk each block and visually check welds, mark defects with paint pen, and write rework tickets. Critical welds also receive MT (magnetic particle) or UT (ultrasonic) testing per ABS and KR class rules. Visual inspection alone cannot catch subsurface defects, but the customer's own post-incident review showed that approximately 73% of the defects causing rework in the last 24 months were surface-visible defects (porosity clusters, undercut, crack initiation, overlap) that should have been caught at visual inspection stage.

### 1.1 Illustrative baseline assumptions

For planning discussion, the fictional scenario assigns the following invented baseline values:

| Metric | Baseline |
|---|---|
| Average block inspection time | 71 minutes |
| Inspection coverage (fraction of weld meterage actually examined) | ~68% |
| Defects detected at inspection | 4.1 per block avg |
| Defects found later (downstream rework or NCR) | 0.9 per block avg |
| Implied miss rate (all classes) | ~18% |
| Implied miss rate (Class-2 defects: porosity, undercut) | ~24% |
| Mean time from weld completion to defect flagged | 4.7 hours |
| Rework cost per downstream defect | USD 480-1,200 |

The 18% miss rate is the high-level figure that triggered the project. Yard-K's head of QA was explicit that they did not want to replace CWIs; they wanted to raise coverage and shrink the feedback loop from "hours" to "seconds."

### 1.2 Why an ML visual inspection system

Three options were considered:

1. **Add more CWIs.** Rejected on cost and because the region has limited availability of qualified welding inspectors (labor capacity is tight; similar shipyards in the area cannot staff the second shift).
2. **Automated radiographic/ultrasonic testing.** Rejected as primary solution because coverage cost is high and it misses surface-only defects that are the main contributor to the miss rate.
3. **Vision-based ML inspection at the welding station itself.** Chosen: catches surface defects at the point and moment of welding, augments rather than replaces CWIs, and provides a structured data feed for process improvement.

The ML system's role was explicitly framed as a **first-pass screening layer** that feeds a **triage queue** reviewed by the CWI. Any weld flagged as "Crack" by the model or flagged as Class-2 defect with confidence above a threshold is escalated. The model's purpose is to raise coverage and shrink the detection-to-flag latency, not to make the final QA call.

---

## 2. Pilot design

### 2.1 Scope

The pilot was scoped to one welding station in the block assembly shop: **Robot Station R-7**, which performs SAW longitudinal welds on stiffener-to-plate assemblies. Two cameras were added:

- **Camera 1 (trailing camera):** mounted on the robot end-effector, 420 mm behind the torch, tilted 22 degrees forward, capturing the cooling weld bead after the slag had been wire-brushed off by a trailing slag removal wheel.
- **Camera 2 (station overview):** fixed camera mounted on a gantry arm 1.8 m above the work surface, 640x640 cropped ROI centered on the weld path.

Both cameras: Basler ace2 Pro a2A2448 (5 MP global shutter), GigE Vision, fixed 12 mm lens. Illumination: two polarized LED bar lights flanking the weld path to suppress arc glare reflections on the plate surface. Trigger mechanism: a PLC tag driven by the robot's "arc off" signal, plus a periodic 2 Hz sample during the weld itself (for process monitoring data, not for defect classification).

### 2.2 Architecture

```
                           ┌──────────────────────┐
                           │ Robot Controller      │
                           │ (FANUC R-30iB)       │
                           │ Siemens S7-1500 PLC  │
                           └────────┬─────────────┘
                                    │ OPC-UA (arc_off, weld_id)
                                    ▼
  ┌─────────┐  GigE   ┌─────────────────────────────────┐
  │ Camera 1│────────▶│                                 │
  └─────────┘         │   Jetson Orin AGX (edge node)   │
                      │   ├─ Image capture daemon       │
  ┌─────────┐  GigE   │   ├─ Triton Inference Server    │
  │ Camera 2│────────▶│   │    (weld_defect.onnx, INT8) │
  └─────────┘         │   ├─ OPC-UA client (asyncua)    │
                      │   └─ MQTT publisher (paho)      │
                      └──────────────┬──────────────────┘
                                     │ MQTT (detections)
                                     ▼
                      ┌──────────────────────────────────┐
                      │ Plant MES (Andon + rework queue)│
                      │ Nexus-Hive analytics ingestion  │
                      └──────────────────────────────────┘
```

The inference service runs on the Jetson Orin AGX 64 GB developer kit (JetPack 6.1). Triton serves the YOLOv8 model exported to ONNX and then compiled to a TensorRT engine with INT8 calibration. The capture daemon is a Python process using `asyncua` to subscribe to the PLC tag and `aiohttp`-style gRPC call to Triton. Detections are published to the plant MQTT broker (Mosquitto on a hardened plant VM) on topics under `yardk/r7/defects/*`.

The MES integration runs on the plant VM and translates MQTT events to:
- an Andon tower light (orange flash) when a Class-2 defect is flagged
- a ticket in the CWI's inspection queue with weld id, camera image, bbox overlay, class, confidence
- a row in the plant historian

The Nexus-Hive analytics stack (see [`Related Projects`](#related-projects)) consumes the same MQTT stream for longitudinal analysis: defect rate by shift, by operator, by wire lot, by consumable batch.

### 2.3 Model choice

The starting point was the base YOLOv8n from this repo. During pilot Yard-K supplied 4,820 labeled weld images from their own QA archive (painted bead photos, spanning 14 months, primarily iPhone and GoPro captures from CWIs). Labeling was done in CVAT by Yard-K QA engineers using the 5 classes from `src/config.py`: crack, porosity, spatter, undercut, overlap.

Fine-tuning was run on a single A100 (rented from a local cloud provider) using the pipeline in `src/train.py`. The scenario compares two hypothetical model-size profiles; these timings were not benchmarked:

- **YOLOv8n**: 3.2M params, smaller, ~8 ms on Orin INT8
- **YOLOv8s**: 11.2M params, ~18 ms on Orin INT8

YOLOv8s was selected for production because the mAP@50 improvement on Crack and Undercut was material (+4.2 mAP@50 points on Crack, +3.1 on Undercut); the latency budget could absorb the cost.

### 2.4 Dataset handling

The Yard-K archive was heavily imbalanced:

| Class | Images | Boxes |
|---|---|---|
| Crack | 181 | 201 |
| Porosity | 2,107 | 4,812 |
| Spatter | 3,301 | 9,104 |
| Undercut | 640 | 740 |
| Overlap | 412 | 468 |

Porosity and Spatter are common and numerous; Crack is rare. We used Yard-K's prior classification records to confirm that this distribution matches the physical defect frequency. Class imbalance was addressed by:

1. Oversampling Crack and Overlap images at training time (class-aware sampler replacing the default).
2. Increased mosaic probability for Crack-containing images (keeping YOLOv8's built-in mosaic on 1.0 for the full set, and adding a second mosaic pass for rare-class images).
3. Loss weighting: `cls_pw` adjusted so per-class loss weight is inversely proportional to sqrt(class frequency).
4. Evaluating per-class mAP separately and gating production promotion on Crack recall, not on aggregate mAP.

---

## 3. 90-day rollout

### Week 1-2: discovery and baseline

- Walked the block assembly floor with the QA manager and a lead welder. Identified Station R-7 as the pilot site because it is the highest-throughput SAW cell and the welds are straight runs (easier for the initial vision problem; curved and corner welds would come later).
- Examined the last 12 months of NCRs (non-conformance reports) and extracted defect class distribution.
- Established baseline metrics: miss rate 18%, mean time-to-flag 4.7 hours, inspection coverage 68%.
- Identified the OPC-UA tag namespace exposed by the Siemens PLC and confirmed read access from a test VM.

### Week 3-4: camera install and image capture

- Installed Camera 1 on the robot end-effector; Camera 2 on the gantry arm. Ran cable trays and power, coordinated with the plant electrician for PoE and GigE Vision routing.
- Captured approximately 8,200 images across 12 shifts with no inference, just logging. This became the production holdout (later labeled).
- Tuned illumination: initial LED bar light angle produced specular glare on the weld bead that obscured porosity clusters. Moved lights to 60-degree angle from the plane of the weld, added polarizers, verified SNR in glare-prone regions.

### Week 5-7: dataset labeling and model training

- Sampled 4,820 historical images from Yard-K's QA archive. Trained a YOLOv8s from `yolov8s.pt` for 120 epochs on A100. mAP@50 at end of training:

| Class | mAP@50 | Precision | Recall |
|---|---|---|---|
| Crack | 0.81 | 0.78 | 0.72 |
| Porosity | 0.89 | 0.86 | 0.88 |
| Spatter | 0.92 | 0.93 | 0.91 |
| Undercut | 0.76 | 0.74 | 0.70 |
| Overlap | 0.68 | 0.71 | 0.63 |
| **Aggregate** | **0.81** | **0.80** | **0.77** |

- Crack recall (0.72) was the gating criterion. Below 0.80 we would not ship; Yard-K's QA head set this bar. We iterated twice: class-aware sampler, then a second round of Crack labels (200 additional crack images pulled from the newly captured week 3-4 pool) raised Crack recall to 0.84.

### Week 8: edge deployment

- Exported `best.pt` to ONNX (opset 17) via `serving/export_onnx.py`. Verified numerical parity against PyTorch on a 50-image spot check (max per-box IoU delta 0.003).
- In the fictional rollout, TensorRT INT8 calibration is assigned an illustrative mAP@50 change from 0.81 (FP32) to 0.79 (INT8), with Crack 0.81 → 0.79 and Overlap 0.68 → 0.65. These values are fabricated, not measured regressions.
- The fictional rollout uses the following invented latency assumptions; no Triton/Orin run produced them:

| Precision | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| FP32 | 42 | 49 | 58 |
| FP16 | 22 | 27 | 31 |
| INT8 | 8.2 | 10.1 | 12.4 |

INT8 selected for production. Latency budget per weld segment (from arc-off trigger to Andon signal) was set at 500 ms. Observed end-to-end: capture to MQTT publish around 180 ms median.

### Week 9: integration with MES and Andon

- Wrote MQTT → MES webhook bridge. MES ticket creation verified.
- Wired Andon tower to the Class-2-flagged topic via the plant PLC (separate PLC from the robot's; this is the plant-floor Andon PLC). PLC pulse on MQTT event triggers the orange flasher.
- Escalation flow: any "crack" class with confidence above 0.60 triggers red light (senior CWI immediate review). Class-2 with confidence above 0.45 triggers orange (queue for next CWI sweep). All detections, regardless of class, are logged to the inspection queue and the historian.

### Week 10-11: shadow mode

- Ran the system in shadow mode for two weeks: detections logged and ticketed, but Andon not wired. CWIs reviewed every ticket and marked it as "agree" or "false positive" in the MES. This produced 1,140 reviewed detections from which we computed:

| Metric | Value |
|---|---|
| CWI agreement (true positive rate) | 87.3% |
| False positive rate (per ticket) | 12.7% |
| False negative rate (CWI found defect not in model's output) | 6.1% |

- The 12.7% false positive rate was higher than the offline test. Root cause analysis identified two drivers: spatter adjacent to porosity was being double-labeled, and a seam between plates with a slight color discontinuity was being flagged as crack. We did not retrain during the pilot; instead we added a simple post-processing rule: suppress a Crack detection if its bbox centroid lies within 6 mm of a known seam (seam locations are in the part geometry). This dropped FPR to 8.4%.

### Week 12-13: cutover to live mode

- Cutover Monday of week 12. Andon wired. CWIs briefed.
- First week post-cutover: 412 blocks inspected, 1,708 detections, 2 crack escalations (one confirmed, one false positive). No missed cracks found by CWI follow-up.
- Second week: added Camera 2 overview feed to the CWI's tablet so they can see the context image when reviewing a ticket. CWI review time dropped from 52 seconds per ticket to 18 seconds.

---

## 4. Illustrative projected outcomes (fictional 90-day scenario)

### 4.1 Fabricated planning values

| Metric | Baseline | Post-deployment | Delta |
|---|---|---|---|
| Miss rate (all classes, estimated) | 18% | 7.1% | -10.9 pts |
| Miss rate (Class-2: porosity, undercut) | 24% | 9.4% | -14.6 pts |
| Mean time from weld to flagged defect | 4.7 hours | 43 seconds | -99.7% |
| Average rework cost per block | USD 780 | USD 410 | -47% |
| Inspection coverage (fraction examined) | 68% | 96% | +28 pts |
| Model P95 inference latency | — | 10.1 ms | — |
| End-to-end latency (arc-off to Andon) | — | ~180 ms | — |

### 4.2 Fabricated per-class scenario values

For the fictional first-30-day narrative, the scenario assigns the following invented values; no CWI labels or live operation produced them:

| Class | Recall | Precision | Confidence threshold |
|---|---|---|---|
| Crack | 0.87 | 0.73 | 0.60 |
| Porosity | 0.91 | 0.84 | 0.45 |
| Spatter | 0.94 | 0.89 | 0.35 |
| Undercut | 0.82 | 0.79 | 0.50 |
| Overlap | 0.71 | 0.68 | 0.55 |

Crack precision (0.73) is lower than recall, as designed. We preferred false escalation to missed crack. Overlap remains the weakest class; retraining after more data accumulates is planned.

### 4.3 Failure modes observed

1. **Torch head vibration on direction changes.** Camera 1 image blur during robot repositioning produced occasional false Spatter detections. Fixed by gating inference on the robot's "stationary-for-100ms" signal.
2. **Temperature sensitivity.** During summer shifts in weeks 14-15 the Orin throttled under 65 degC ambient. Installed a deflector and added an additional case fan; confirmed operation stable at 41 degC internal at 38 degC ambient.
3. **Wire lot drift.** In week 9 post-cutover, a new consumable lot produced a subtly different bead sheen. Porosity detection precision dropped from 0.84 to 0.71 over the next week. Detected via the monitoring pipeline in [`docs/production/monitoring-drift.md`](../production/monitoring-drift.md). Triggered a retraining cycle against recent labeled data.

---

## 5. Integration details

### 5.1 PLC signal → API call flow

The Siemens S7-1500 exposes OPC-UA. The robot controller writes to two tags:

- `Robot.ArcOff` — pulses True when the arc extinguishes.
- `Robot.WeldID` — a string with format `B{block}-S{segment}-{timestamp}` set at arc-on.

The edge capture daemon (see `integrations/opc-ua/client.py` for a similar reference implementation) subscribes to `Robot.ArcOff`:

```
on Robot.ArcOff rising edge:
    weld_id = read(Robot.WeldID)
    images = capture_from_cameras(cam1, cam2, count=3, spacing_ms=120)
    for image in images:
        detections = triton_infer(image)
        if detections:
            publish_mqtt(topic=f"yardk/r7/defects/{weld_id}",
                         payload={weld_id, image_ref, detections})
```

Three images per weld are captured over a 360 ms window to handle the bead cooling and allow the slag-clearing wheel to fully pass. The latest of the three is used for the primary detection; all three are archived.

The MES ingest service subscribes to the MQTT topic and creates an inspection ticket. The Andon tower is driven by a separate plant-floor PLC that also subscribes via an MQTT-to-Profinet bridge.

### 5.2 Operator UX

The CWI carries a ruggedized 10-inch tablet running a small React app that displays:

- The inspection queue (all open tickets, sorted by severity and age).
- For each ticket: the annotated image, the class, the confidence, the weld id, the block and segment, and two buttons: `Confirm` and `Mark false positive`.
- A "escalate to senior welder" button for cracks; senior welder is dispatched by pager.

CWIs retain full authority. If a CWI marks a ticket as false positive, the MES records the disagreement and the image is routed into the retraining candidates queue. Over the first 30 days, 1,407 tickets went through this flow, of which 201 were marked false positive and queued for retraining.

### 5.3 Escalation policy

| Trigger | Action | Owner |
|---|---|---|
| Crack class, conf >= 0.60 | Red Andon + CWI dispatch + senior welder pager | Senior CWI |
| Crack class, conf 0.45-0.60 | Orange Andon + CWI ticket | CWI |
| Class-2 cluster (>=3 porosity boxes in one image) | Orange Andon + rework candidate flag | CWI |
| Undercut, conf >= 0.50, bbox area > 12 mm2 equivalent | Orange Andon + ticket | CWI |
| Any other detection | Log-only ticket | CWI batch review |

The thresholds were set through a joint workshop with the Yard-K QA head, the Senior CWI, and the line supervisor. The explicit framing: "each threshold is a bet about false positive cost vs false negative cost; we can retune."

---

## 6. Lessons learned

### 6.1 Dataset

The archive-first approach worked but was slower than ideal. Two-thirds of the labeling effort went into the initial 4,820 images; after cutover the human-in-the-loop labeling pipeline (see [`docs/production/labeling-pipeline.md`](../production/labeling-pipeline.md)) started producing fresh labels at roughly 80-120 per day, which is enough to sustain retraining cycles.

For future deployments we will front-load capture-then-label for two weeks before training begins. The archive images captured the defect classes but did not reflect the specific camera geometry and illumination of the pilot station, and fine-tuning on those images required an additional domain-adaptation step.

### 6.2 Edge thermal

Jetson Orin thermal headroom on a production floor is tighter than the datasheet suggests once the enclosure is closed. Plan for active cooling with margin for the hottest expected shift (Korean summer or equivalent). See [`docs/production/edge-deployment.md`](../production/edge-deployment.md) for the thermal mitigation pattern.

### 6.3 Model drift

The consumable-lot drift event was the single most important lesson of the pilot. We had monitoring in place (confidence histogram drift + periodic re-evaluation against a labeled holdout) but the signal was noisy at the resolution we were looking at (daily). We moved to shift-level (8-hour window) aggregates and added a per-class precision-on-confirmed-TP metric as a faster-responding indicator. See [`docs/production/monitoring-drift.md`](../production/monitoring-drift.md).

### 6.4 CWI buy-in

The CWIs were initially concerned that the system was being introduced to replace them. We addressed this early by framing the system as a coverage multiplier ("you are now inspecting 96% of welds instead of 68%") and by ensuring every model output routed through them. By week 6 post-cutover, the lead CWI was the most active advocate for expanding the system to Station R-4 and R-9.

### 6.5 False positive cost

A false positive in this setting is not free. Every false positive that gets escalated to "crack" is a welder who has to stop, a senior welder paged to review, and a block moved in the schedule. The FPR of 12.7% in shadow mode would have been unacceptable in live mode; the 8.4% after the seam-suppression rule was the minimum acceptable figure for cutover. Class-level precision thresholds are the main lever; we set them conservatively and loosened them only as confidence in specific classes grew.

### 6.6 Gauge R&R

The fictional scenario includes a notional Gauge R&R exercise with fabricated agreement values (91%, 84%, and 86%) to show what a real study might report. No welds or appraisers were evaluated, so these numbers establish neither model consistency nor human-human variability.

### 6.7 FMEA update

Yard-K's FMEA (Failure Mode and Effects Analysis) was updated to add a row for "Vision system false negative on crack class." Detection mitigation is the existing CWI sweep plus the downstream MT/UT spot check. Severity, occurrence, and detection scores were rated collaboratively with the QA head. The vision system's contribution was captured as reducing the `Occurrence` rating for classes 1-4 by 1 (porosity, undercut, spatter, overlap); `Crack` was left unchanged pending additional data.

---

## 7. What did not work

In the spirit of honesty:

- **First torch-mounted camera was too close.** 180 mm from the torch; the arc glare during welding saturated the sensor and even with bandpass filtering we could not recover usable frames during the arc. Moved to 420 mm and switched to post-weld-only capture.
- **Initial attempt to run inference on every frame during welding.** Threw compute budget and battery at it; no useful signal because the bead is still glowing. Abandoned. Inference is triggered only after arc-off.
- **Tried a segmentation model (YOLOv8-seg) first.** Gave up after 10 days because the box model was sufficient for the routing decision (which class, where) and the segmentation labels were an order of magnitude more work to produce. Box model shipped.
- **Considered on-device LLM for explanation text.** Rejected as premature; a fixed template for the ticket reason is sufficient and the senior CWI prefers structured output.

---

## 8. What would be next (roadmap, not yet implemented)

- **Add Station R-4** (pulsed GMAW positional welds, harder visual domain) as the second site. Expected model fine-tune rather than greenfield training.
- **Expand to subsurface defects** by fusing the vision model output with the UT sensor data at the same station. Two-sensor late-fusion rather than end-to-end model.
- **Closed-loop parameter adjustment.** If the model detects rising porosity rate in a shift, automatically adjust shielding gas flow rate within a predefined envelope. Requires change control with the welding engineer; 6-month horizon.
- **Per-operator performance review.** Only with explicit agreement from labor representatives; this is an ethics and HR question before it is a technical one. See [`governance/ethics-architecture.md`](../../governance/ethics-architecture.md).

---

## 9. Stakeholder list

| Role | Organization | Responsibility |
|---|---|---|
| Head of QA | Yard-K | Sponsor, owns the decision on threshold setting |
| Lead CWI | Yard-K | Primary user of the ticket queue; authority on final calls |
| Senior Welder (afternoon shift) | Yard-K | Pager recipient for crack escalations |
| Welding Engineer | Yard-K | Updates WPS (welding procedure spec) when drift is traced to process |
| Plant IT | Yard-K | Manages the edge node, MQTT broker, MES ingest |
| Robotics Integrator | Third-party vendor | Maintains the FANUC R-30iB; coordinated camera install |
| Solutions Architect | External (this project) | Pilot design, model training, integration glue |
| Data Engineer | External | Labeling pipeline, retraining cadence |

---

## 10. Cross-references in this repo

- The detection model and training pipeline: [`src/train.py`](../../src/train.py), [`src/config.py`](../../src/config.py).
- Triton serving config: [`serving/triton/model_repository/weld_defect/config.pbtxt`](../../serving/triton/model_repository/weld_defect/config.pbtxt).
- Edge deployment: [`edge/jetson-orin/`](../../edge/jetson-orin/).
- OPC-UA client reference: [`integrations/opc-ua/client.py`](../../integrations/opc-ua/client.py).
- MQTT publisher reference: [`integrations/mqtt/publisher.py`](../../integrations/mqtt/publisher.py).
- Model card: [`governance/model-card.md`](../../governance/model-card.md).
- Datasheet: [`governance/data-sheet.md`](../../governance/data-sheet.md).

## 11. Related projects

This deployment is described here in isolation, but in a real plant the ML inspection system is one node in a wider observability and ops graph:

- **[AegisOps](https://github.com/KIM3310/AegisOps)** — the operator handoff and incident analysis pattern used in this case study for CWI shift changes and escalation postmortems.
- **[Nexus-Hive](https://github.com/KIM3310/Nexus-Hive)** — the analytics layer consuming the MQTT defect telemetry for longitudinal analysis (defect rate by shift, by operator, by wire lot).
- **[retina-scan-ai](https://github.com/KIM3310/retina-scan-ai)** — sibling vision repo (medical imaging); same core patterns, different regulatory envelope.
