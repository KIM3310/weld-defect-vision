"""MQTT publisher for weld defect detections.

Wraps paho-mqtt with sensible defaults for plant-floor deployments:
    - TLS 1.2+ with optional client certs
    - Auto-reconnect with backoff
    - QoS 1 for detection publishes (at-least-once); QoS 0 for heartbeats
    - Sparkplug-B-compatible topic layout (configurable)

Usage:
    # As a module, from the OPC-UA client:
    from integrations.mqtt.publisher import WeldDefectPublisher
    pub = WeldDefectPublisher.from_env()
    pub.publish_detection(weld_id="B41-S128-...", detections=[...], severity="high")

    # Directly, for a demo loop:
    python integrations/mqtt/publisher.py --demo --broker localhost --count 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import ssl
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("weld-mqtt-publisher")


@dataclass
class MqttConfig:
    broker: str = field(default_factory=lambda: os.getenv("MQTT_BROKER", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("MQTT_PORT", "1883")))
    tls: bool = field(default_factory=lambda: os.getenv("MQTT_TLS", "false").lower() == "true")
    tls_ca_cert: str = field(default_factory=lambda: os.getenv("MQTT_TLS_CA_CERT", ""))
    tls_client_cert: str = field(default_factory=lambda: os.getenv("MQTT_TLS_CLIENT_CERT", ""))
    tls_client_key: str = field(default_factory=lambda: os.getenv("MQTT_TLS_CLIENT_KEY", ""))
    username: str = field(default_factory=lambda: os.getenv("MQTT_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("MQTT_PASSWORD", ""))
    topic_prefix: str = field(default_factory=lambda: os.getenv("MQTT_TOPIC_PREFIX", "plant/r7"))
    keepalive: int = field(default_factory=lambda: int(os.getenv("MQTT_KEEPALIVE", "30")))
    client_id: str = field(
        default_factory=lambda: os.getenv(
            "MQTT_CLIENT_ID",
            f"weld-defect-{uuid.uuid4().hex[:8]}",
        )
    )


class WeldDefectPublisher:
    def __init__(self, cfg: MqttConfig) -> None:
        self.cfg = cfg
        self.connected = False
        self._configure_client()

    @classmethod
    def from_env(cls) -> "WeldDefectPublisher":
        return cls(MqttConfig())

    def _configure_client(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise ImportError(
                "paho-mqtt is not installed. Install with: pip install paho-mqtt"
            ) from exc

        self.client = mqtt.Client(
            client_id=self.cfg.client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv5,
        )

        if self.cfg.username:
            self.client.username_pw_set(self.cfg.username, self.cfg.password)

        if self.cfg.tls:
            tls_args: dict[str, Any] = {"tls_version": ssl.PROTOCOL_TLSv1_2}
            if self.cfg.tls_ca_cert:
                tls_args["ca_certs"] = self.cfg.tls_ca_cert
            if self.cfg.tls_client_cert and self.cfg.tls_client_key:
                tls_args["certfile"] = self.cfg.tls_client_cert
                tls_args["keyfile"] = self.cfg.tls_client_key
            self.client.tls_set(**tls_args)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)

        self.client.connect(self.cfg.broker, self.cfg.port, keepalive=self.cfg.keepalive)
        self.client.loop_start()

    def _on_connect(self, _client: Any, _userdata: Any, _flags: Any, reason_code: Any, _props: Any = None) -> None:
        self.connected = True
        log.info("MQTT connected (rc=%s) to %s:%s", reason_code, self.cfg.broker, self.cfg.port)

    def _on_disconnect(self, _client: Any, _userdata: Any, _flags: Any, reason_code: Any, _props: Any = None) -> None:
        self.connected = False
        log.warning("MQTT disconnected (rc=%s)", reason_code)

    def publish_detection(
        self,
        weld_id: str,
        station: str,
        detections: list[dict],
        severity: str,
        model_version: str,
        image_ref: str | None = None,
    ) -> None:
        """Publish a detection event to plant/<prefix>/defects/<severity>/<weld_id>."""
        topic = f"{self.cfg.topic_prefix}/defects/{severity}/{weld_id}"
        payload = {
            "ts": time.time(),
            "station": station,
            "weld_id": weld_id,
            "severity": severity,
            "model_version": model_version,
            "image_ref": image_ref,
            "detections": detections,
        }
        self.client.publish(topic, json.dumps(payload), qos=1)

    def publish_heartbeat(self, station: str, payload: dict | None = None) -> None:
        topic = f"{self.cfg.topic_prefix}/heartbeat"
        body = {"ts": time.time(), "station": station}
        if payload:
            body.update(payload)
        self.client.publish(topic, json.dumps(body), qos=0)

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as exc:  # noqa: BLE001
            log.warning("MQTT close failed: %s", exc)


def _demo(broker: str, count: int) -> int:
    os.environ.setdefault("MQTT_BROKER", broker)
    pub = WeldDefectPublisher.from_env()

    stop = False

    def _signal(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)

    try:
        for i in range(count):
            if stop:
                break
            weld_id = f"DEMO-R7-{i:05d}"
            detections = [
                {
                    "class_id": 1,
                    "class_name": "Porosity",
                    "confidence": 0.81,
                    "bbox": [120, 88, 240, 202],
                }
            ]
            pub.publish_detection(
                weld_id=weld_id,
                station="R7",
                detections=detections,
                severity="low",
                model_version="weld_defect_v7.3.1_int8",
                image_ref=f"s3://demo/{weld_id}.jpg",
            )
            log.info("published demo event %s", weld_id)
            time.sleep(0.5)
    finally:
        pub.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--broker", type=str, default="localhost")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.demo:
        return _demo(args.broker, args.count)

    sys.stderr.write("No action requested. Use --demo to run a demo loop.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
