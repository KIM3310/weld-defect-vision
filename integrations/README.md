# MES / SCADA / Broker Integrations

Reference implementations for the four canonical patterns used to integrate the weld-defect inference service with plant-floor systems. See [`docs/production/mes-scada-integration.md`](../docs/production/mes-scada-integration.md) for the full rationale and topology.

## Layout

```
integrations/
├── README.md
├── opc-ua/
│   └── client.py            asyncua-based OPC-UA client (PLC signal -> inference trigger)
├── mqtt/
│   └── publisher.py         paho-mqtt based publisher for plant broker
├── rest-webhook/
│   └── webhook_sender.py    httpx-based retrying webhook sender for MES
└── kafka/
    └── producer.py          aiokafka producer for analytics backbone
```

## When to use each

| Upstream / downstream | Pattern | Reference |
|---|---|---|
| Robot controller / welding machine PLC | OPC-UA | `opc-ua/client.py` |
| Andon PLC (writeback) | OPC-UA | same client; uses `write_value` |
| Plant-wide SCADA / HMI | MQTT | `mqtt/publisher.py` |
| Historian / time-series DB | MQTT (via a bridge) | `mqtt/publisher.py` |
| MES rework queue | REST webhook | `rest-webhook/webhook_sender.py` |
| Analytics / data lake | Kafka | `kafka/producer.py` |

## Environment variables

The reference code reads configuration from environment variables for deployability. Defaults are sensible for development:

```
OPCUA_URL=opc.tcp://plc.plant.internal:4840
OPCUA_USER=
OPCUA_PASSWORD=
OPCUA_ARC_OFF_NODE=ns=3;s=Robot.ArcOff
OPCUA_WELD_ID_NODE=ns=3;s=Robot.WeldID

MQTT_BROKER=mqtt.plant.internal
MQTT_PORT=1883
MQTT_TLS=true
MQTT_USER=
MQTT_PASSWORD=
MQTT_TOPIC_PREFIX=plant/r7

WEBHOOK_URL=https://mes.plant.internal/api/v1/rework/tickets
WEBHOOK_TOKEN=

KAFKA_BOOTSTRAP=kafka.plant.internal:9092
KAFKA_TOPIC=plant.biw.defects

TRITON_URL=localhost:8001
```

## Testing against local mocks

Each client has an env-var-controlled endpoint; point at a local dev broker / server:

```bash
# MQTT: run Mosquitto locally
docker run -p 1883:1883 eclipse-mosquitto:2
MQTT_BROKER=localhost python -m integrations.mqtt.publisher --demo

# Kafka: run Redpanda locally
docker run -p 9092:9092 redpandadata/redpanda start --smp 1
KAFKA_BOOTSTRAP=localhost:9092 python -m integrations.kafka.producer --demo
```

## Security

See [`docs/production/mes-scada-integration.md#7-security`](../docs/production/mes-scada-integration.md). Short version:

- OPC-UA: SignAndEncrypt with X.509.
- MQTT: TLS 1.2+ with client certs + broker ACLs.
- Kafka: SASL_SSL with client certs + topic-level ACLs.
- Webhook: mTLS + short-lived bearer tokens.
