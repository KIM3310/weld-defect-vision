# ADR 003: Edge inference on Jetson Orin; cloud reserved for training and batch analytics

- **Status**: Accepted
- **Date**: 2026-03-28
- **Deciders**: ML owner; Plant IT owner (customer, shipyard)

## Context

A welding station produces image data that must yield a decision within a budget of 150-500 ms (depending on the deployment — shorter for automotive, longer for shipyard). The choices are:

- Run inference at the edge on hardware adjacent to the camera.
- Run inference in a plant-local datacenter or in the cloud.

Network to the cloud is a 100 Mbps symmetric link to a regional AWS region; plant-local data center has a 1 Gbps uplink.

## Decision

**Inference runs at the edge** on a Jetson Orin AGX adjacent to the welding cell. Training, batch analytics, and model artifact management run in cloud / plant datacenter tiers.

## Options considered

### A. Fully cloud inference

Pros:
- No edge hardware to manage.
- Easier model updates (push to one server, not N edge nodes).
- GPU capacity can scale to new stations without per-station hardware purchase.

Cons:
- **Round-trip latency.** From the plant to AWS ap-northeast-2 (Seoul region) measured at 18-35 ms network RTT. Add TLS + HTTP overhead + queueing = 60-120 ms per request. For automotive 250 ms per-ROI budget this is half the budget gone before compute starts.
- **Reliability**: any WAN flap stops the line. Plants expect per-station-local availability; WAN uptime is not a QA input.
- **Confidentiality**: weld images leaving the plant perimeter is a customer concern.
- **Cost**: egress bandwidth for continuous image streams is non-trivial.

Rejected.

### B. Plant-local datacenter inference

Pros:
- Low network latency (2-5 ms).
- No WAN dependency for the hot path.
- Centralized compute; simpler ops than per-station edge.

Cons:
- Still a network hop per inference (vs zero on edge).
- Rack space and power in the datacenter may not be available.
- Plant datacenters in older plants are not GPU-friendly environments.
- For the automotive case the plant datacenter is fine; for the shipyard the datacenter is 400 m away across the dock and the cable plant does not support GPU-density racks.

Chosen for automotive. Rejected for shipyard.

### C. Edge inference (Jetson Orin per station)

Pros:
- Zero network latency for the hot path (camera -> inference is all local).
- WAN-independent operation; an internet outage does not stop the line.
- Inference results are local-first; detection telemetry goes to the plant broker over the LAN, not the WAN.
- Power envelope is tractable (~30W) compared to a rack GPU (~250W+).

Cons:
- Per-station hardware cost (~USD 2,000 per Orin + cameras).
- Fleet management: N edge nodes means N systemd units, N model updates, N thermal sensors to monitor.
- Compute ceiling lower than a bench GPU; multi-model pipelines have to fit in the Orin budget.
- Thermal management on the plant floor is a real concern (see shipyard case study).

Chosen for shipyard (primary). Chosen secondarily for automotive stations that need per-station compute (e.g. future Phase 2 with per-station model updates).

### D. Hybrid: edge primary with cloud fallback

Evaluated. Rejected because the cloud fallback is only useful if the edge fails; if the edge is down the sensible response is to fall back to the pre-ML manual inspection path (human CWI walking the weld), not to a cloud path that still requires the edge's camera to be accessible (which it may not be if the edge is down for camera-adjacent reasons).

## Rationale

Edge inference maps onto the operational requirements: low latency, WAN-independent availability, data locality. The per-station hardware cost is easily justified against rework savings in both case studies.

The automotive case study uses a centralized line-local server (2x L4 GPU) rather than per-station Orin because the 6-camera station fits the centralized pattern better: dynamic batching across cameras, one server to maintain, one model to update. This is still "edge" in the WAN-sense (on-prem, LAN-adjacent), just not per-station.

## Consequences

### Positive

- Sub-10 ms model latency on Orin + ~180 ms end-to-end means generous headroom for post-processing and MQTT publish.
- Plant floor failure modes (WAN, upstream services) do not stop the line.
- Detection data stays inside the plant perimeter by default.

### Negative

- Fleet management is the main ongoing cost. Mitigations:
  - Standard systemd unit across all edge nodes.
  - Watchdog service on each edge node.
  - Centralized log aggregation (journald -> Loki or equivalent).
  - Model updates via a plant-local artifact registry pull.

## Cloud-side responsibilities (non-hot-path)

- **Training**: runs in a rented GPU environment (A100 or H100); artifacts published to the model registry.
- **Model registry**: S3-compatible storage, versioned.
- **Batch analytics**: Nexus-Hive consumes the Kafka / MQTT firehose.
- **Long-term archive**: images older than 14 days roll from the edge NVMe to plant S3.

## References

- Shipyard edge Jetson deployment: [`docs/case-studies/shipyard-pipeline.md`](../case-studies/shipyard-pipeline.md).
- Automotive centralized station server: [`docs/case-studies/automotive-body-shop.md`](../case-studies/automotive-body-shop.md).
- Edge deployment doc: [`docs/production/edge-deployment.md`](../production/edge-deployment.md).
- Latency benchmarks: [`benchmarks/results/cpu-vs-gpu-vs-jetson.json`](../../benchmarks/results/cpu-vs-gpu-vs-jetson.json).

## Revision history

- 2026-03-28: Initial decision.
- 2026-04-12: Added automotive-centralized-station clarification.
