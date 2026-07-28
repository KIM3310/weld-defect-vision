"""Kafka producer for weld defect detection events.

Designed for the automotive case study where detection events feed the
plant analytics backbone (consumed by Nexus-Hive or equivalent). Key
decisions:
    - aiokafka for asyncio-friendly produce
    - JSON serialization (swap for Avro when a schema registry is in use)
    - Partition key on body_id / weld_id so per-body events are ordered
    - Idempotent producer with enable_idempotence=True
    - Compression=snappy for throughput

Usage:
    from integrations.kafka.producer import WeldDefectKafkaProducer

    prod = WeldDefectKafkaProducer.from_env()
    await prod.start()
    await prod.send_detection({...})
    await prod.close()

    # Demo loop:
    python integrations/kafka/producer.py --demo --bootstrap localhost:9092 --count 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("weld-kafka-producer")


@dataclass
class KafkaConfig:
    bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    )
    topic: str = field(default_factory=lambda: os.getenv("KAFKA_TOPIC", "plant.biw.defects"))
    security_protocol: str = field(
        default_factory=lambda: os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    )
    sasl_mechanism: str = field(
        default_factory=lambda: os.getenv("KAFKA_SASL_MECHANISM", "")
    )
    sasl_username: str = field(default_factory=lambda: os.getenv("KAFKA_SASL_USERNAME", ""))
    sasl_password: str = field(default_factory=lambda: os.getenv("KAFKA_SASL_PASSWORD", ""))
    ssl_cafile: str = field(default_factory=lambda: os.getenv("KAFKA_SSL_CAFILE", ""))
    ssl_certfile: str = field(default_factory=lambda: os.getenv("KAFKA_SSL_CERTFILE", ""))
    ssl_keyfile: str = field(default_factory=lambda: os.getenv("KAFKA_SSL_KEYFILE", ""))
    client_id: str = field(default_factory=lambda: os.getenv("KAFKA_CLIENT_ID", "weld-defect-producer"))


class WeldDefectKafkaProducer:
    def __init__(self, cfg: KafkaConfig) -> None:
        self.cfg = cfg
        self._producer: Any = None

    @classmethod
    def from_env(cls) -> WeldDefectKafkaProducer:
        return cls(KafkaConfig())

    async def start(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer
        except ImportError as exc:
            raise ImportError(
                "aiokafka is not installed. Install with: pip install aiokafka"
            ) from exc

        kwargs: dict[str, Any] = {
            "bootstrap_servers": self.cfg.bootstrap_servers,
            "client_id": self.cfg.client_id,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "key_serializer": lambda k: k.encode("utf-8") if k else None,
            "enable_idempotence": True,
            "compression_type": "snappy",
            "linger_ms": 5,
            "max_batch_size": 32 * 1024,
            "acks": "all",
        }

        if self.cfg.security_protocol != "PLAINTEXT":
            kwargs["security_protocol"] = self.cfg.security_protocol
            if self.cfg.sasl_mechanism:
                kwargs["sasl_mechanism"] = self.cfg.sasl_mechanism
                kwargs["sasl_plain_username"] = self.cfg.sasl_username
                kwargs["sasl_plain_password"] = self.cfg.sasl_password
            if self.cfg.ssl_cafile:
                import ssl

                context = ssl.create_default_context(cafile=self.cfg.ssl_cafile)
                if self.cfg.ssl_certfile and self.cfg.ssl_keyfile:
                    context.load_cert_chain(self.cfg.ssl_certfile, self.cfg.ssl_keyfile)
                kwargs["ssl_context"] = context

        self._producer = AIOKafkaProducer(**kwargs)
        await self._producer.start()
        log.info(
            "Kafka producer started (bootstrap=%s topic=%s)",
            self.cfg.bootstrap_servers,
            self.cfg.topic,
        )

    async def send_detection(
        self,
        body_id: str | None,
        weld_id: str,
        station: str,
        detections: list[dict],
        severity: str,
        decision: str,
        model_version: str,
        body_model: str | None = None,
    ) -> None:
        if self._producer is None:
            raise RuntimeError("Call start() before send_detection().")

        key = body_id or weld_id
        payload = {
            "body_id": body_id,
            "weld_id": weld_id,
            "station": station,
            "body_model": body_model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "decision": decision,
            "severity": severity,
            "model_version": model_version,
            "rois": [
                {
                    "detections": detections,
                }
            ],
        }

        try:
            await self._producer.send_and_wait(
                topic=self.cfg.topic,
                value=payload,
                key=key,
                headers=[
                    ("schema_version", b"1"),
                    ("source", b"weld-defect-vision"),
                ],
            )
        except Exception as exc:
            log.error("kafka produce failed weld_id=%s: %s", weld_id, exc)
            raise

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
            log.info("Kafka producer stopped.")


async def _demo(bootstrap: str, count: int, topic: str) -> int:
    os.environ["KAFKA_BOOTSTRAP"] = bootstrap
    os.environ["KAFKA_TOPIC"] = topic
    prod = WeldDefectKafkaProducer.from_env()
    await prod.start()
    try:
        for i in range(count):
            body_id = f"Z24-2026-WK16-{i:04d}"
            await prod.send_detection(
                body_id=body_id,
                weld_id=f"{body_id}-weld-1",
                station="WQ",
                detections=[
                    {
                        "class_id": 1,
                        "class_name": "Porosity",
                        "confidence": 0.81,
                        "bbox": [120, 88, 240, 202],
                    }
                ],
                severity="medium",
                decision="rework",
                model_version="weld_defect_v7.3.1_fp16",
                body_model="sedan_Z24_ph2",
            )
            log.info("sent %s", body_id)
            await asyncio.sleep(0.05)
    finally:
        await prod.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--bootstrap", type=str, default="localhost:9092")
    parser.add_argument("--topic", type=str, default="plant.biw.defects")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.demo:
        return asyncio.run(_demo(args.bootstrap, args.count, args.topic))

    sys.stderr.write("No action requested. Use --demo to run a demo loop.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
