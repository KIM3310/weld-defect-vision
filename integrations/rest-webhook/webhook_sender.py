"""REST webhook sender for MES rework-ticket creation.

Posts weld defect detection events to a configured MES endpoint. Designed
for production reliability:
    - Idempotency key: weld_id + detection_timestamp
    - Exponential backoff with jitter on transient failures
    - Local disk-backed dead-letter queue if MES is unreachable > N minutes
    - Bearer token + optional mTLS

Usage:
    from integrations.rest_webhook.webhook_sender import WebhookSender

    sender = WebhookSender.from_env()
    await sender.send(
        weld_id="B41-S128-20260416T144512",
        station="R7",
        defect_class="crack",
        severity="high",
        confidence=0.82,
        image_url="https://plant-s3.internal/...",
    )

    # Demo:
    python integrations/rest-webhook/webhook_sender.py --demo --url http://localhost:8080/hook
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("weld-webhook-sender")


@dataclass
class WebhookConfig:
    url: str = field(default_factory=lambda: os.getenv("WEBHOOK_URL", "http://localhost:8080/hook"))
    token: str = field(default_factory=lambda: os.getenv("WEBHOOK_TOKEN", ""))
    ca_cert: str = field(default_factory=lambda: os.getenv("WEBHOOK_CA_CERT", ""))
    client_cert: str = field(default_factory=lambda: os.getenv("WEBHOOK_CLIENT_CERT", ""))
    client_key: str = field(default_factory=lambda: os.getenv("WEBHOOK_CLIENT_KEY", ""))

    timeout_s: float = field(default_factory=lambda: float(os.getenv("WEBHOOK_TIMEOUT_S", "5.0")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("WEBHOOK_MAX_RETRIES", "6")))
    backoff_base_s: float = field(default_factory=lambda: float(os.getenv("WEBHOOK_BACKOFF_BASE_S", "0.5")))
    backoff_max_s: float = field(default_factory=lambda: float(os.getenv("WEBHOOK_BACKOFF_MAX_S", "30.0")))

    dlq_dir: str = field(default_factory=lambda: os.getenv("WEBHOOK_DLQ_DIR", "/var/lib/weld-defect/dlq"))
    dlq_after_minutes: int = field(default_factory=lambda: int(os.getenv("WEBHOOK_DLQ_AFTER_MINUTES", "10")))


class WebhookSender:
    def __init__(self, cfg: WebhookConfig) -> None:
        self.cfg = cfg
        self._outage_started: float | None = None
        self._client: Any = None

    @classmethod
    def from_env(cls) -> WebhookSender:
        return cls(WebhookConfig())

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is not installed. Install with: pip install httpx"
            ) from exc

        headers = {"Content-Type": "application/json"}
        if self.cfg.token:
            headers["Authorization"] = f"Bearer {self.cfg.token}"

        verify: Any = True
        if self.cfg.ca_cert:
            verify = self.cfg.ca_cert

        cert: Any = None
        if self.cfg.client_cert and self.cfg.client_key:
            cert = (self.cfg.client_cert, self.cfg.client_key)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.cfg.timeout_s),
            headers=headers,
            verify=verify,
            cert=cert,
        )
        return self._client

    def _idempotency_key(self, weld_id: str, detection_timestamp: str) -> str:
        return f"{weld_id}:{detection_timestamp}"

    async def send(
        self,
        weld_id: str,
        station: str,
        defect_class: str,
        severity: str,
        confidence: float,
        image_url: str | None,
        body_id: str | None = None,
        model_version: str = "weld_defect_v0",
    ) -> bool:
        detection_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        payload = {
            "source": f"weld-defect-vision-{station}",
            "weld_id": weld_id,
            "body_id": body_id,
            "station": station,
            "defect_class": defect_class,
            "severity": severity,
            "confidence": confidence,
            "image_url": image_url,
            "detection_timestamp": detection_timestamp,
            "model_version": model_version,
            "_idempotency_key": self._idempotency_key(weld_id, detection_timestamp),
        }
        return await self._send_with_retry(payload)

    async def _send_with_retry(self, payload: dict) -> bool:
        client = self._ensure_client()
        attempt = 0
        while attempt <= self.cfg.max_retries:
            try:
                resp = await client.post(
                    self.cfg.url,
                    json=payload,
                    headers={"X-Idempotency-Key": payload["_idempotency_key"]},
                )
                if 200 <= resp.status_code < 300:
                    self._outage_started = None
                    log.info("MES webhook ok weld_id=%s", payload["weld_id"])
                    return True
                if resp.status_code in {408, 429} or 500 <= resp.status_code < 600:
                    log.warning(
                        "MES webhook transient status=%s body=%s",
                        resp.status_code,
                        resp.text[:200],
                    )
                else:
                    log.error(
                        "MES webhook permanent status=%s body=%s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return False
            except Exception as exc:  # noqa: BLE001
                log.warning("MES webhook attempt %s failed: %s", attempt, exc)

            if self._outage_started is None:
                self._outage_started = time.time()
            if self._outage_elapsed_minutes() >= self.cfg.dlq_after_minutes:
                self._write_dlq(payload)
                return False

            sleep = min(
                self.cfg.backoff_max_s,
                self.cfg.backoff_base_s * (2 ** attempt) * random.uniform(0.8, 1.2),
            )
            await asyncio.sleep(sleep)
            attempt += 1

        self._write_dlq(payload)
        return False

    def _outage_elapsed_minutes(self) -> float:
        if self._outage_started is None:
            return 0.0
        return (time.time() - self._outage_started) / 60.0

    def _write_dlq(self, payload: dict) -> None:
        try:
            out_dir = Path(self.cfg.dlq_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}.json"
            out_path = out_dir / fname
            out_path.write_text(json.dumps(payload, indent=2))
            log.warning("DLQ: wrote unsent webhook payload to %s", out_path)
        except Exception as exc:  # noqa: BLE001
            log.error("DLQ write failed: %s", exc)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


async def _demo(url: str, count: int) -> int:
    os.environ["WEBHOOK_URL"] = url
    sender = WebhookSender.from_env()
    try:
        for i in range(count):
            ok = await sender.send(
                weld_id=f"DEMO-R7-{i:05d}",
                station="R7",
                defect_class="porosity",
                severity="medium",
                confidence=0.81,
                image_url="https://plant-s3.internal/demo.jpg",
                model_version="weld_defect_v7.3.1_int8",
            )
            log.info("sent=%s ok=%s", i, ok)
            await asyncio.sleep(0.2)
    finally:
        await sender.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--url", type=str, default="http://localhost:8080/hook")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.demo:
        return asyncio.run(_demo(args.url, args.count))

    sys.stderr.write("No action requested. Use --demo to run a demo.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
