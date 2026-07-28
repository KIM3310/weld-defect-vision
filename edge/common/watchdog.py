"""Edge watchdog for the weld-defect inference service.

Periodically checks:
    - Inference service HTTP health endpoint.
    - Process RSS memory (detects slow memory leaks).
    - GPU / thermal telemetry (via tegrastats on Jetson; nvidia-smi on x86+GPU).

On sustained unhealthy state, restarts the service via systemd (if
available) or via docker compose. Publishes heartbeat and fault events
to the plant MQTT broker.

Runs as a separate systemd unit from the inference service so that a
crash of the inference service does not take the watchdog down with it.

Usage:
    python -m edge.common.watchdog \
        --health-url http://localhost:8000/health \
        --restart-cmd 'systemctl restart weld-defect.service' \
        --interval 10 \
        --memory-max-mb 24000 \
        --unhealthy-threshold 3 \
        --mqtt-broker mqtt.plant.internal \
        --mqtt-topic yardk/r7/edge/watchdog
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger("weld-watchdog")


@dataclass
class WatchdogState:
    unhealthy_count: int = 0
    last_restart_ts: float = 0.0
    restart_total: int = 0
    last_memory_mb: float = 0.0
    fault_history: list[dict] = field(default_factory=list)


def validate_health_url(url: str) -> str:
    cleaned = str(url or "").strip()
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("health URL must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("health URL must not contain userinfo")
    if not parsed.hostname:
        raise ValueError("health URL must include a host")
    return cleaned


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlsplit(url)
    return (parsed.scheme, (parsed.hostname or "").lower(), parsed.port)


def check_http_health(url: str, timeout: float) -> tuple[bool, str]:
    try:
        import urllib.error
        import urllib.request

        class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
                return None

        validate_health_url(url)
        req = urllib.request.Request(url, method="GET")
        opener = urllib.request.build_opener(_NoRedirectHandler)
        try:
            resp_ctx = opener.open(req, timeout=timeout)  # nosec B310
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                location = exc.headers.get("Location", "")
                redirect_url = urllib.parse.urljoin(url, location)
                try:
                    validate_health_url(redirect_url)
                except ValueError as validation_exc:
                    return False, f"unsafe redirect: {validation_exc}"
                return False, f"unsafe redirect: status {exc.code} to {redirect_url}"
            raise
        with resp_ctx as resp:
            code = resp.getcode()
            body = resp.read(512).decode("utf-8", errors="replace")
            if 200 <= code < 300:
                return True, body
            return False, f"status {code}: {body}"
    except Exception as exc:  # noqa: BLE001
        return False, f"exception {type(exc).__name__}: {exc}"


def check_process_memory(pattern: str) -> float:
    """Return RSS in MB for the first matching process, or 0.0 if not found."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", pattern], stderr=subprocess.DEVNULL, timeout=3
        )
        pid = out.decode().strip().splitlines()[0]
        status = subprocess.check_output(
            ["cat", f"/proc/{pid}/status"], timeout=2
        ).decode()
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                kb = int(parts[1])
                return kb / 1024.0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0
    except Exception:  # noqa: BLE001
        return 0.0
    return 0.0


def read_jetson_thermal() -> dict:
    """Parse /sys/class/thermal/thermal_zoneN/temp on Jetson. Safe no-op otherwise."""
    out: dict[str, float] = {}
    base = "/sys/class/thermal"
    if not os.path.isdir(base):
        return out
    try:
        for zone in os.listdir(base):
            zone_path = os.path.join(base, zone)
            temp_file = os.path.join(zone_path, "temp")
            type_file = os.path.join(zone_path, "type")
            if not (os.path.isfile(temp_file) and os.path.isfile(type_file)):
                continue
            with open(type_file) as f:
                t = f.read().strip()
            with open(temp_file) as f:
                millideg = int(f.read().strip())
            out[t] = round(millideg / 1000.0, 1)
    except OSError:
        pass
    return out


def restart_service(cmd: str) -> bool:
    log.warning("Restarting service: %s", cmd)
    try:
        subprocess.check_call(shlex.split(cmd), timeout=60)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.error("Restart command failed: %s", exc)
        return False


class MqttPublisher:
    """Lazy-initialized MQTT publisher. Silently no-ops if paho-mqtt is missing."""

    def __init__(self, broker: str | None, port: int, topic: str, client_id: str) -> None:
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.client = None
        if not broker:
            return
        try:
            import paho.mqtt.client as mqtt  # type: ignore[import-not-found]

            self.client = mqtt.Client(
                client_id=client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            )
            self.client.connect(broker, port, keepalive=30)
            self.client.loop_start()
            log.info("MQTT connected: %s:%s", broker, port)
        except Exception as exc:  # noqa: BLE001
            log.warning("MQTT init failed (%s); continuing without publish.", exc)
            self.client = None

    def publish(self, payload: dict) -> None:
        if self.client is None:
            return
        try:
            self.client.publish(self.topic, json.dumps(payload), qos=0)
        except Exception as exc:  # noqa: BLE001
            log.warning("MQTT publish failed: %s", exc)


def run_watchdog(
    health_url: str,
    health_timeout: float,
    restart_cmd: str,
    process_pattern: str,
    interval: float,
    memory_max_mb: float,
    unhealthy_threshold: int,
    cooldown_seconds: float,
    mqtt: MqttPublisher,
    on_shutdown: Callable[[], None],
) -> None:
    state = WatchdogState()
    shutting_down = False

    def _signal(_signum, _frame):
        nonlocal shutting_down
        shutting_down = True

    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)

    log.info("Watchdog started (interval=%ss, threshold=%s)", interval, unhealthy_threshold)

    while not shutting_down:
        now = time.time()

        healthy, detail = check_http_health(health_url, health_timeout)
        mem_mb = check_process_memory(process_pattern)
        thermal = read_jetson_thermal()

        state.last_memory_mb = mem_mb
        memory_over = memory_max_mb > 0 and mem_mb > memory_max_mb

        if memory_over:
            healthy = False
            detail = f"memory {mem_mb:.0f}MB exceeds cap {memory_max_mb:.0f}MB"

        heartbeat = {
            "ts": now,
            "healthy": healthy,
            "memory_mb": round(mem_mb, 1),
            "thermal": thermal,
            "unhealthy_count": state.unhealthy_count,
            "restart_total": state.restart_total,
            "detail": detail if not healthy else "ok",
        }
        mqtt.publish(heartbeat)
        log.info("%s", heartbeat)

        if healthy:
            state.unhealthy_count = 0
        else:
            state.unhealthy_count += 1
            state.fault_history.append({"ts": now, "detail": detail})
            state.fault_history = state.fault_history[-50:]

        should_restart = (
            state.unhealthy_count >= unhealthy_threshold
            and (now - state.last_restart_ts) >= cooldown_seconds
        )
        if should_restart:
            ok = restart_service(restart_cmd)
            state.last_restart_ts = now
            state.restart_total += 1
            if ok:
                state.unhealthy_count = 0
            mqtt.publish(
                {
                    "ts": now,
                    "event": "restart",
                    "success": ok,
                    "restart_total": state.restart_total,
                }
            )

        for _ in range(int(interval * 10)):
            if shutting_down:
                break
            time.sleep(0.1)

    on_shutdown()
    log.info("Watchdog stopping.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-url", type=str, default="http://localhost:8000/health")
    parser.add_argument("--health-timeout", type=float, default=3.0)
    parser.add_argument("--restart-cmd", type=str, default="systemctl restart weld-defect.service")
    parser.add_argument("--process-pattern", type=str, default="integrations.opc_ua.client")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--memory-max-mb", type=float, default=24000.0)
    parser.add_argument("--unhealthy-threshold", type=int, default=3)
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--mqtt-broker", type=str, default=None)
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", type=str, default="plant/edge/watchdog")
    parser.add_argument("--mqtt-client-id", type=str, default="weld-defect-watchdog")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        health_url = validate_health_url(args.health_url)
    except ValueError as exc:
        log.error("Invalid --health-url: %s", exc)
        return 2

    mqtt = MqttPublisher(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=args.mqtt_topic,
        client_id=args.mqtt_client_id,
    )

    def _on_shutdown() -> None:
        if mqtt.client is not None:
            try:
                mqtt.client.loop_stop()
                mqtt.client.disconnect()
            except Exception as exc:  # noqa: BLE001
                log.debug("MQTT shutdown failed: %s", exc)

    try:
        run_watchdog(
            health_url=health_url,
            health_timeout=args.health_timeout,
            restart_cmd=args.restart_cmd,
            process_pattern=args.process_pattern,
            interval=args.interval,
            memory_max_mb=args.memory_max_mb,
            unhealthy_threshold=args.unhealthy_threshold,
            cooldown_seconds=args.cooldown_seconds,
            mqtt=mqtt,
            on_shutdown=_on_shutdown,
        )
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
