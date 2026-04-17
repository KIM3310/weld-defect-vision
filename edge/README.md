# Edge Deployment

Artifacts for deploying `weld-defect-vision` on an edge compute node at a welding station.

## Supported targets

- **Jetson Orin AGX 64 GB** (primary reference). See [`jetson-orin/`](./jetson-orin/).
- **Jetson Orin NX 16 GB** (supported; same container and systemd unit, lower per-instance throughput).
- **x86 + NVIDIA L4** (used in centralized line-side deployments; see the automotive case study).

## Layout

```
edge/
├── README.md
├── jetson-orin/
│   ├── Dockerfile                    Jetson container image
│   ├── setup.sh                      Device bootstrap (JetPack check, Docker, deps, model pull)
│   └── systemd/
│       └── weld-defect.service       systemd unit for the inference service
└── common/
    └── watchdog.py                   Periodic health check that restarts on memory leak / crash
```

## Bootstrap flow (Jetson Orin)

```bash
# 1. Flash JetPack 6.1 on the Orin via NVIDIA SDK Manager (off-device step).

# 2. On the Orin, pull this repo and run the setup script:
curl -fsSL https://raw.githubusercontent.com/KIM3310/weld-defect-vision/main/edge/jetson-orin/setup.sh | bash

# 3. Deploy model artifacts:
#    - Push the TensorRT engine (compiled on this device) to:
#      /opt/weld-defect-vision/models/weld_defect/1/model.plan
#    - Push the model config:
#      /opt/weld-defect-vision/models/weld_defect/config.pbtxt

# 4. Enable and start the service:
sudo systemctl enable --now weld-defect.service

# 5. Verify:
journalctl -u weld-defect.service -f
curl http://localhost:8000/health
```

See [`docs/production/edge-deployment.md`](../docs/production/edge-deployment.md) for the full runbook, including thermal tuning and networking.

## Watchdog

[`common/watchdog.py`](./common/watchdog.py) runs as a second service on the edge node, independent of the inference service. It pings the health endpoint, checks process memory, and emits MQTT heartbeats. On sustained unhealthy state it restarts the inference service via systemd.

## Related

- Shipyard case study uses this layout: [`docs/case-studies/shipyard-pipeline.md`](../docs/case-studies/shipyard-pipeline.md).
- Integration with plant infrastructure: [`integrations/`](../integrations/).
- Benchmarks on Jetson: [`benchmarks/results/cpu-vs-gpu-vs-jetson.json`](../benchmarks/results/cpu-vs-gpu-vs-jetson.json).
