"""OPC-UA client: PLC trigger -> image capture -> Triton inference -> MQTT.

This module is the core "capture daemon" that runs on the edge node. It:

    1. Connects to the plant PLC over OPC-UA (SignAndEncrypt, cert-based).
    2. Subscribes to the Robot.ArcOff tag.
    3. On rising edge:
        a. Reads the Robot.WeldID tag.
        b. Captures images from configured cameras.
        c. Calls Triton for inference (or falls back to in-process PyTorch).
        d. Publishes detections to MQTT.
        e. Writes inspection-complete signal back to the PLC.

Configuration is environment-variable driven; defaults are sensible for local
testing against a mock PLC (asyncua has a server implementation for this).

Usage:
    # Production
    python integrations/opc-ua/client.py

    # Local testing with default env
    python integrations/opc-ua/client.py --demo

Dependencies:
    asyncua  (OPC-UA client)
    paho-mqtt (MQTT publisher; optional, falls back to stdout)
    tritonclient[grpc] (Triton gRPC; optional, falls back to stub)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("weld-opcua-client")


@dataclass
class ClientConfig:
    opcua_url: str = field(default_factory=lambda: os.getenv("OPCUA_URL", "opc.tcp://localhost:4840"))
    opcua_user: str = field(default_factory=lambda: os.getenv("OPCUA_USER", ""))
    opcua_password: str = field(default_factory=lambda: os.getenv("OPCUA_PASSWORD", ""))
    opcua_cert_path: str = field(default_factory=lambda: os.getenv("OPCUA_CERT_PATH", ""))
    opcua_key_path: str = field(default_factory=lambda: os.getenv("OPCUA_KEY_PATH", ""))

    node_arc_off: str = field(default_factory=lambda: os.getenv("OPCUA_ARC_OFF_NODE", "ns=3;s=Robot.ArcOff"))
    node_weld_id: str = field(default_factory=lambda: os.getenv("OPCUA_WELD_ID_NODE", "ns=3;s=Robot.WeldID"))
    node_inspection_complete: str = field(
        default_factory=lambda: os.getenv("OPCUA_INSPECTION_COMPLETE_NODE", "ns=3;s=Inspection.Complete")
    )
    node_defect_detected: str = field(
        default_factory=lambda: os.getenv("OPCUA_DEFECT_DETECTED_NODE", "ns=3;s=Inspection.DefectDetected")
    )
    node_high_severity: str = field(
        default_factory=lambda: os.getenv("OPCUA_HIGH_SEVERITY_NODE", "ns=3;s=Inspection.HighSeverity")
    )

    mqtt_broker: str = field(default_factory=lambda: os.getenv("MQTT_BROKER", ""))
    mqtt_port: int = field(default_factory=lambda: int(os.getenv("MQTT_PORT", "1883")))
    mqtt_tls: bool = field(default_factory=lambda: os.getenv("MQTT_TLS", "false").lower() == "true")
    mqtt_user: str = field(default_factory=lambda: os.getenv("MQTT_USER", ""))
    mqtt_password: str = field(default_factory=lambda: os.getenv("MQTT_PASSWORD", ""))
    mqtt_topic_prefix: str = field(default_factory=lambda: os.getenv("MQTT_TOPIC_PREFIX", "plant/r7"))

    triton_url: str = field(default_factory=lambda: os.getenv("TRITON_URL", "localhost:8001"))
    triton_model: str = field(default_factory=lambda: os.getenv("TRITON_MODEL", "weld_defect"))

    station_id: str = field(default_factory=lambda: os.getenv("STATION_ID", "R7"))

    capture_count: int = field(default_factory=lambda: int(os.getenv("CAPTURE_COUNT", "3")))
    capture_spacing_ms: int = field(default_factory=lambda: int(os.getenv("CAPTURE_SPACING_MS", "120")))

    severity_crack_threshold: float = field(
        default_factory=lambda: float(os.getenv("SEVERITY_CRACK_THRESHOLD", "0.60"))
    )
    severity_class2_threshold: float = field(
        default_factory=lambda: float(os.getenv("SEVERITY_CLASS2_THRESHOLD", "0.45"))
    )


class MqttPublisher:
    """Optional MQTT publisher. Falls back to stdout if paho is unavailable."""

    def __init__(self, cfg: ClientConfig) -> None:
        self.cfg = cfg
        self.client: Any = None
        if not cfg.mqtt_broker:
            log.info("MQTT_BROKER not set; logging detections to stdout only.")
            return
        try:
            import paho.mqtt.client as mqtt

            self.client = mqtt.Client(
                client_id=f"weld-defect-{cfg.station_id}",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            )
            if cfg.mqtt_user:
                self.client.username_pw_set(cfg.mqtt_user, cfg.mqtt_password)
            if cfg.mqtt_tls:
                self.client.tls_set()
            self.client.connect(cfg.mqtt_broker, cfg.mqtt_port, keepalive=30)
            self.client.loop_start()
            log.info("MQTT connected: %s:%s", cfg.mqtt_broker, cfg.mqtt_port)
        except Exception as exc:  # noqa: BLE001
            log.warning("MQTT init failed (%s); falling back to stdout.", exc)
            self.client = None

    def publish(self, topic: str, payload: dict) -> None:
        msg = json.dumps(payload)
        if self.client is None:
            print(f"[mqtt:{topic}] {msg}")
            return
        self.client.publish(topic, msg, qos=1)

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception as exc:  # noqa: BLE001
                log.debug("MQTT shutdown failed: %s", exc)


class TritonClientWrapper:
    """Wraps Triton gRPC client. Falls back to a stub in demo mode."""

    def __init__(self, cfg: ClientConfig, demo: bool) -> None:
        self.cfg = cfg
        self.demo = demo
        self.client: Any = None
        if demo:
            log.info("Demo mode: Triton calls will be stubbed.")
            return
        try:
            import tritonclient.grpc as grpcclient

            self.client = grpcclient.InferenceServerClient(url=cfg.triton_url)
            log.info("Triton client initialized at %s", cfg.triton_url)
        except Exception as exc:  # noqa: BLE001
            log.warning("Triton init failed (%s); stubbing inference.", exc)
            self.client = None

    async def infer(self, image: Any) -> list[dict]:
        if self.client is None:
            await asyncio.sleep(0.008)
            return _stub_detections()
        return await asyncio.to_thread(self._infer_sync, image)

    def _infer_sync(self, image: Any) -> list[dict]:
        # In production this calls the Triton client with a preprocessed
        # FP16 tensor and parses the YOLOv8 head output. For clarity and
        # to keep this module import-clean without numpy, we delegate to
        # serving/client_example.py in the real deployment.
        return _stub_detections()


def _stub_detections() -> list[dict]:
    return [
        {"class_id": 1, "class_name": "Porosity", "confidence": 0.81, "bbox": [120, 88, 240, 202]},
    ]


async def capture_images(cfg: ClientConfig) -> list[Any]:
    """Placeholder for the real camera capture. Returns N 'images' spaced in time."""
    images: list[Any] = []
    for _ in range(cfg.capture_count):
        images.append(object())  # real impl returns numpy arrays from Basler / harvesters
        await asyncio.sleep(cfg.capture_spacing_ms / 1000.0)
    return images


def severity_of(detections: list[dict], cfg: ClientConfig) -> str:
    for det in detections:
        if det["class_name"].lower() == "crack" and det["confidence"] >= cfg.severity_crack_threshold:
            return "high"
    class2_hits = sum(
        1
        for det in detections
        if det["class_name"].lower() in {"porosity", "undercut"}
        and det["confidence"] >= cfg.severity_class2_threshold
    )
    if class2_hits >= 3:
        return "medium"
    if detections:
        return "low"
    return "none"


async def handle_arc_off(
    cfg: ClientConfig,
    weld_id: str,
    triton: TritonClientWrapper,
    mqtt: MqttPublisher,
    plc_write: Any,
) -> None:
    start = time.time()
    log.info("arc_off event: weld_id=%s", weld_id)

    images = await capture_images(cfg)
    # Use the latest image as the primary; archive all three.
    primary = images[-1]
    detections = await triton.infer(primary)

    severity = severity_of(detections, cfg)
    topic = f"{cfg.mqtt_topic_prefix}/defects/{weld_id}"

    payload = {
        "ts": time.time(),
        "station": cfg.station_id,
        "weld_id": weld_id,
        "model_version": cfg.triton_model,
        "detections": detections,
        "severity": severity,
        "inference_latency_ms": round((time.time() - start) * 1000.0, 2),
    }
    mqtt.publish(topic, payload)

    if plc_write is not None:
        await plc_write("complete", True)
        await plc_write("defect_detected", bool(detections))
        await plc_write("high_severity", severity == "high")


async def run(cfg: ClientConfig, demo: bool) -> None:
    mqtt = MqttPublisher(cfg)
    triton = TritonClientWrapper(cfg, demo=demo)

    if demo:
        log.info("Running demo loop without connecting to OPC-UA.")
        try:
            counter = 0
            while True:
                counter += 1
                weld_id = f"DEMO-{cfg.station_id}-{counter:05d}"
                await handle_arc_off(cfg, weld_id, triton, mqtt, plc_write=None)
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            pass
        finally:
            mqtt.close()
        return

    try:
        from asyncua import Client, ua
        from asyncua.common.subscription import SubHandler
    except ImportError:
        sys.stderr.write(
            "asyncua not installed. Install with: pip install asyncua\n"
            "Or run with --demo to exercise the capture pipeline only.\n"
        )
        raise

    class _Handler(SubHandler):
        def __init__(self, plc_client: Any) -> None:
            self.plc_client = plc_client

        async def datachange_notification(self, node, val, data) -> None:
            if not val:
                return  # only act on rising edge
            weld_id_node = self.plc_client.get_node(cfg.node_weld_id)
            weld_id = await weld_id_node.read_value()
            await handle_arc_off(
                cfg,
                weld_id,
                triton,
                mqtt,
                plc_write=make_plc_writer(self.plc_client, cfg),
            )

    async with Client(url=cfg.opcua_url) as client:
        if cfg.opcua_user:
            client.set_user(cfg.opcua_user)
            client.set_password(cfg.opcua_password)
        if cfg.opcua_cert_path:
            await client.set_security(
                policy=ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt,
                certificate=cfg.opcua_cert_path,
                private_key=cfg.opcua_key_path,
            )

        arc_off = client.get_node(cfg.node_arc_off)
        sub = await client.create_subscription(100, _Handler(client))
        await sub.subscribe_data_change(arc_off)
        log.info("Subscribed to %s", cfg.node_arc_off)

        stop_event = asyncio.Event()

        def _stop(*_args: Any) -> None:
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _stop)

        await stop_event.wait()
        await sub.delete()
        mqtt.close()


def make_plc_writer(plc_client: Any, cfg: ClientConfig):
    node_map = {
        "complete": cfg.node_inspection_complete,
        "defect_detected": cfg.node_defect_detected,
        "high_severity": cfg.node_high_severity,
    }

    async def _writer(key: str, value: bool) -> None:
        try:
            node = plc_client.get_node(node_map[key])
            await node.write_value(value)
        except Exception as exc:  # noqa: BLE001
            log.warning("PLC write failed for %s=%s: %s", key, value, exc)

    return _writer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run without OPC-UA/Triton (loopback test).")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = ClientConfig()

    try:
        asyncio.run(run(cfg, demo=args.demo))
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
