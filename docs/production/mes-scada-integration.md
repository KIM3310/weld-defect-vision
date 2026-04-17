# MES / SCADA Integration

The defect-detection service integrates with plant-floor systems through four canonical patterns: OPC-UA (to PLC), MQTT (to SCADA and plant broker), Kafka (to analytics backbone), and REST webhook (to MES for ticket creation). This document covers the patterns, the rationale, and the reference implementations in [`integrations/`](../../integrations/).

---

## 1. Which pattern for which system

| Upstream / downstream | Typical system | Pattern | Rationale |
|---|---|---|---|
| Robot controller / welding machine PLC | Siemens S7, FANUC R-30iB, Mitsubishi MELSEC | OPC-UA | Universal industrial protocol; most controllers expose OPC-UA natively or via a small gateway |
| Plant-wide SCADA / HMI | Ignition, WinCC, Factory Talk | MQTT (Sparkplug B optional) | Pub/sub topology; many SCADA suites have native MQTT connectors |
| Andon tower / line-stop interlock | Plant-floor PLC | OPC-UA tag write | Same controller or a dedicated Andon PLC; write-enabled OPC-UA tag |
| MES (work order, rework queue, traceability) | SAP ME, MES Apriso, custom | REST webhook | MES vendors prefer HTTPS/JSON for external events |
| Analytics / historian / data lake | Kafka → Snowflake / Databricks / Nexus-Hive | Kafka produce | Replayable, versioned stream; analytics consumes at its own pace |

The shipyard case uses OPC-UA (inbound) + MQTT (outbound to plant broker). The automotive case uses OPC-UA (Andon) + Kafka (analytics) + REST webhook (rework ticket).

---

## 2. OPC-UA: PLC signal → API call flow

OPC-UA is the trigger mechanism. The robot controller publishes a tag that pulses True when the arc extinguishes. The edge daemon subscribes to that tag and, on rising edge, kicks off the image capture and inference sequence.

### 2.1 Canonical flow

```
Robot Controller
  │
  ▼
[PLC writes tag: Robot.ArcOff = True]
  │
  ▼                                         Edge daemon
OPC-UA subscription                          (asyncua, Python)
  │                                            │
  └── rising edge ─────────────────────────────▶
                                              │
                          ┌───────────────────┘
                          │
                          ▼
                      Capture 3 images from camera
                          │
                          ▼
                      Call Triton (gRPC localhost)
                          │
                          ▼
                      Publish to MQTT topic (detections)
                          │
                          ▼
                      Write result tag back to PLC
                      (Robot.InspectionComplete = True)
```

### 2.2 asyncua reference

See [`integrations/opc-ua/client.py`](../../integrations/opc-ua/client.py) for the full async subscription pattern. Outline:

```python
from asyncua import Client
from asyncua.common.subscription import SubHandler

class ArcOffHandler(SubHandler):
    async def datachange_notification(self, node, val, data):
        if val:
            await handle_arc_off_event()

async with Client(url="opc.tcp://plc.internal:4840") as client:
    arc_off_node = client.get_node("ns=3;s=Robot.ArcOff")
    sub = await client.create_subscription(100, ArcOffHandler())
    await sub.subscribe_data_change(arc_off_node)
    await asyncio.Future()  # run forever
```

### 2.3 Authentication

OPC-UA supports three security modes: None, Sign, SignAndEncrypt. Production deployments use SignAndEncrypt with X.509 certificates issued by the plant CA. The asyncua client accepts a cert/key pair; see the client reference for the production config.

### 2.4 Tag namespace

Define the namespace early with plant controls engineers. Example agreed namespace for the shipyard case:

| Tag | Direction | Type | Meaning |
|---|---|---|---|
| `Robot.ArcOn` | Subscribe | BOOL | Pulses True at arc strike |
| `Robot.ArcOff` | Subscribe | BOOL | Pulses True at arc extinguish |
| `Robot.WeldID` | Read on event | STRING | Unique identifier for this weld |
| `Robot.WeldProgram` | Read on event | STRING | Welding program id |
| `Inspection.Ready` | Write | BOOL | True while edge daemon is healthy |
| `Inspection.Complete` | Write | BOOL | True after inference + publish |
| `Inspection.DefectDetected` | Write | BOOL | True if any defect flagged |
| `Inspection.HighSeverity` | Write | BOOL | True if crack or Class-2 escalation |

The PLC's reaction to the `HighSeverity` tag (Andon red, welder stop, etc.) is the plant's responsibility and is defined by the welding engineer, not the ML team.

---

## 3. MQTT: plant-broker pub/sub

MQTT is the canonical pub/sub mechanism for plant-floor telemetry. The edge daemon publishes to a broker (Mosquitto, EMQX, HiveMQ) and multiple downstream consumers subscribe:

- The MES ingest service (creates rework ticket).
- The historian / time-series database.
- The Andon bridge (if Andon is not on the same PLC).
- The Nexus-Hive analytics ingestion.

### 3.1 Topic hierarchy

```
plant/<area>/<station>/defects/<class>/<weld_id>
plant/<area>/<station>/heartbeat
plant/<area>/<station>/thermal
plant/<area>/<station>/model_version
```

Example: `yardk/block_assembly/r7/defects/crack/B41-S128-20260416T144512`.

### 3.2 Payload (JSON)

```json
{
  "ts": "2026-04-16T14:45:12.441Z",
  "station": "R7",
  "weld_id": "B41-S128-20260416T144512",
  "image_ref": "s3://yardk-welds/2026/04/16/R7/B41-S128-20260416T144512-cam1-f0.jpg",
  "model_version": "weld_defect_v7.3.1_int8",
  "detections": [
    {"class": "crack", "conf": 0.82, "bbox": [120, 88, 340, 412]}
  ],
  "severity": "high"
}
```

### 3.3 QoS and reliability

- Detection publishes: QoS 1 (at-least-once). Idempotency at the subscriber is based on `weld_id`.
- Heartbeats: QoS 0.
- Broker high-availability: two Mosquitto nodes, shared persistence, client uses multiple connection attempts.

See [`integrations/mqtt/publisher.py`](../../integrations/mqtt/publisher.py) for the reference publisher (paho-mqtt async).

### 3.4 Sparkplug B

When the plant standardizes on Sparkplug B (common in Ignition deployments), wrap the payload in the Sparkplug B protobuf with NBIRTH/NDEATH lifecycle messages. The logic is the same; the wire format differs.

---

## 4. REST webhook: MES ticket creation

MES integration via webhook is operationally the simplest pattern. The edge daemon (or a MQTT→webhook bridge) posts a JSON body to the MES endpoint:

```
POST https://mes.plant.internal/api/v1/rework/tickets
Authorization: Bearer <plant-svc-token>
Content-Type: application/json

{
  "source": "weld-defect-vision-r7",
  "weld_id": "B41-S128-20260416T144512",
  "body_id": null,
  "station": "R7",
  "defect_class": "crack",
  "severity": "high",
  "confidence": 0.82,
  "image_url": "https://plant-s3.internal/weld-archive/2026/04/16/R7/B41-S128-cam1-f0.jpg",
  "detection_timestamp": "2026-04-16T14:45:12.441Z",
  "model_version": "weld_defect_v7.3.1_int8"
}
```

See [`integrations/rest-webhook/webhook_sender.py`](../../integrations/rest-webhook/webhook_sender.py) for the reference sender (httpx with retry).

### 4.1 Idempotency

The MES must treat `weld_id + detection_timestamp` as the idempotency key. The sender retries on network failure; the MES silently drops duplicates.

### 4.2 Delivery guarantees

Webhook senders use exponential backoff + dead-letter queue. If MES is unavailable for > 10 minutes, the sender stages events to local disk and replays on recovery.

---

## 5. Kafka: analytics backbone

Kafka is the downstream pattern for the automotive case study and for any plant that has standardized on Kafka as an event backbone. Characteristics:

- Exactly-once semantics (via transactions) if the plant has configured them.
- 14-30 day retention on the detection topic is typical.
- The Nexus-Hive analytics layer consumes from Kafka.

### 5.1 Topic design

```
plant_c.biw.inspection.defects  (3 partitions, 14-day retention)
plant_c.biw.inspection.heartbeats (1 partition, 2-day retention)
plant_c.biw.inspection.model_events (1 partition, 90-day retention)
```

Partition key: `body_id` (keeps all events for one body in order on one partition).

See [`integrations/kafka/producer.py`](../../integrations/kafka/producer.py) for the reference producer (aiokafka).

### 5.2 Schema management

Use Confluent Schema Registry or equivalent; event schemas are versioned JSON Schema or Avro. Schema compatibility policy: `BACKWARD` (new consumers can read old messages).

---

## 6. End-to-end sequence diagram (shipyard, simplified)

```
Robot     PLC       EdgeDaemon   Triton    MQTT Broker   MES
  │        │            │           │           │          │
  │Weld    │            │           │           │          │
  │arc on  │            │           │           │          │
  │────────▶            │           │           │          │
  │        │ArcOn=True  │           │           │          │
  │        │            │           │           │          │
  │        │ ... welding│           │           │          │
  │        │            │           │           │          │
  │Weld    │            │           │           │          │
  │arc off │            │           │           │          │
  │────────▶            │           │           │          │
  │        │ArcOff=True │           │           │          │
  │        │            │           │           │          │
  │        │────OPC-UA notify──────▶│           │          │
  │        │            │capture 3  │           │          │
  │        │            │images    │            │          │
  │        │            │──────gRPC─▶           │          │
  │        │            │           │infer      │          │
  │        │            │◀──detect──│           │          │
  │        │            │           │           │          │
  │        │            │──────MQTT publish────▶│          │
  │        │            │           │           │─webhook──▶│
  │        │            │           │           │          │create
  │        │            │           │           │          │ticket
  │        │            │           │           │          │
  │        │◀───write InspectionComplete=True───│          │
  │        │                                                │
```

---

## 7. Security

- OPC-UA: SignAndEncrypt with X.509; plant CA issues certs.
- MQTT: TLS 1.2+ with client certs; broker ACL restricts topics per client.
- Kafka: SASL_SSL with client certs or OAUTHBEARER; topic-level ACLs.
- Webhook: mTLS + bearer token (short-lived, rotated by plant identity provider).

Secrets never live on the edge device unencrypted. Use systemd credentials or a plant-local secrets manager.

---

## 8. References

- OPC-UA client: [`integrations/opc-ua/client.py`](../../integrations/opc-ua/client.py).
- MQTT publisher: [`integrations/mqtt/publisher.py`](../../integrations/mqtt/publisher.py).
- Kafka producer: [`integrations/kafka/producer.py`](../../integrations/kafka/producer.py).
- REST webhook: [`integrations/rest-webhook/webhook_sender.py`](../../integrations/rest-webhook/webhook_sender.py).
- Shipyard case study flow: [`docs/case-studies/shipyard-pipeline.md#5-integration-details`](../case-studies/shipyard-pipeline.md).
- Automotive case study flow: [`docs/case-studies/automotive-body-shop.md#8-integration-with-mes-and-warranty-data`](../case-studies/automotive-body-shop.md).
