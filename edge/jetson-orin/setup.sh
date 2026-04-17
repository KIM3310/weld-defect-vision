#!/usr/bin/env bash
# Bootstrap a Jetson Orin for weld-defect-vision.
#
# Idempotent. Safe to re-run. Writes everything under /opt/weld-defect-vision
# and registers a systemd service under /etc/systemd/system.
#
# Tested on: JetPack 6.1 / L4T r36.4 on Orin AGX 64 GB and Orin NX 16 GB.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/KIM3310/weld-defect-vision/main/edge/jetson-orin/setup.sh | bash
#   # OR
#   sudo bash edge/jetson-orin/setup.sh

set -euo pipefail

INSTALL_DIR=${INSTALL_DIR:-/opt/weld-defect-vision}
SERVICE_USER=${SERVICE_USER:-weld-defect}
REPO_URL=${REPO_URL:-https://github.com/KIM3310/weld-defect-vision.git}
REPO_REF=${REPO_REF:-main}
MODEL_URL=${MODEL_URL:-}
ENV_FILE=/etc/weld-defect/env

log() {
    printf '[setup] %s\n' "$*"
}

err() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

require_root() {
    if [ "$(id -u)" -ne 0 ]; then
        err "Run as root or via sudo."
    fi
}

check_jetson() {
    if [ ! -f /etc/nv_tegra_release ]; then
        err "This is not a Jetson device (/etc/nv_tegra_release missing)."
    fi
    log "Detected: $(cat /etc/nv_tegra_release | head -n1)"
}

check_l4t_version() {
    local version
    version=$(grep -oP 'R\d+ \(REVISION: \d+\.\d+' /etc/nv_tegra_release | head -n1 || true)
    log "L4T: ${version:-unknown}"
    if [ -z "${version}" ]; then
        log "Warning: could not parse L4T version; continuing."
    fi
}

install_base_deps() {
    log "Installing base dependencies (apt)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
        jq \
        netcat-openbsd \
        chrony \
        docker.io \
        docker-compose-plugin
    systemctl enable --now docker
    systemctl enable --now chrony
}

enable_nvidia_container_runtime() {
    log "Configuring NVIDIA container runtime"
    local conf=/etc/docker/daemon.json
    mkdir -p /etc/docker
    if [ ! -f "$conf" ]; then
        cat > "$conf" <<'JSON'
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
JSON
        systemctl restart docker
    else
        log "daemon.json already exists; leaving untouched. Verify default-runtime is 'nvidia'."
    fi
}

create_service_user() {
    if id "$SERVICE_USER" >/dev/null 2>&1; then
        log "User $SERVICE_USER already exists"
    else
        log "Creating system user $SERVICE_USER"
        useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
    fi
    usermod -aG docker "$SERVICE_USER" || true
}

fetch_repo() {
    log "Fetching repo into $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    if [ -d "$INSTALL_DIR/.git" ]; then
        git -C "$INSTALL_DIR" fetch --all --prune
        git -C "$INSTALL_DIR" checkout "$REPO_REF"
        git -C "$INSTALL_DIR" pull --ff-only
    else
        git clone --branch "$REPO_REF" "$REPO_URL" "$INSTALL_DIR"
    fi

    mkdir -p \
        "$INSTALL_DIR/logs" \
        "$INSTALL_DIR/image-archive" \
        "$INSTALL_DIR/models/weld_defect/1"
    chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR"
}

pull_model() {
    if [ -z "$MODEL_URL" ]; then
        log "MODEL_URL not set; skipping model pull. Place your engine at"
        log "  $INSTALL_DIR/models/weld_defect/1/model.plan"
        return 0
    fi
    log "Pulling model artifact from $MODEL_URL"
    curl -fsSL "$MODEL_URL" -o "$INSTALL_DIR/models/weld_defect/1/model.plan"
    chown "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR/models/weld_defect/1/model.plan"
}

write_env_file() {
    log "Writing $ENV_FILE"
    mkdir -p "$(dirname "$ENV_FILE")"
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" <<'ENV'
# weld-defect-vision edge environment.
# Rotate secrets out of this file into a plant secrets manager in production.
OPCUA_URL=opc.tcp://plc.plant.internal:4840
MQTT_BROKER=mqtt.plant.internal
MQTT_PORT=1883
TRITON_URL=localhost:8001
CAMERA_1_IP=10.20.30.41
CAMERA_2_IP=10.20.30.42
STATION_ID=R7
LOG_LEVEL=INFO
ENV
        chmod 0640 "$ENV_FILE"
        chown root:"$SERVICE_USER" "$ENV_FILE"
    else
        log "$ENV_FILE already exists; leaving untouched."
    fi
}

install_systemd_unit() {
    log "Installing systemd unit"
    install -m 0644 \
        "$INSTALL_DIR/edge/jetson-orin/systemd/weld-defect.service" \
        /etc/systemd/system/weld-defect.service
    systemctl daemon-reload
    systemctl enable weld-defect.service
    log "Service enabled (not started). Start with: systemctl start weld-defect.service"
}

set_power_mode() {
    # MAXN on AGX is nvpmodel -m 0; on NX it varies.
    if command -v nvpmodel >/dev/null 2>&1; then
        log "Setting nvpmodel to mode 0 (MAXN) for inference workload"
        nvpmodel -m 0 || log "nvpmodel failed (non-fatal)"
    fi
    if command -v jetson_clocks >/dev/null 2>&1; then
        log "Locking clocks at max (jetson_clocks)"
        jetson_clocks || log "jetson_clocks failed (non-fatal)"
    fi
}

summary() {
    cat <<SUMMARY

Setup complete.

Install dir:      $INSTALL_DIR
Service user:     $SERVICE_USER
Environment file: $ENV_FILE
systemd unit:     /etc/systemd/system/weld-defect.service

Next steps:
  1. Edit $ENV_FILE with the plant-specific OPC-UA URL, MQTT broker, cameras.
  2. Place your TensorRT engine at:
       $INSTALL_DIR/models/weld_defect/1/model.plan
     (If you do not have one yet, run serving/export_tensorrt.py on this device.)
  3. Start the service:
       sudo systemctl start weld-defect.service
       journalctl -u weld-defect.service -f

SUMMARY
}

main() {
    require_root
    check_jetson
    check_l4t_version
    install_base_deps
    enable_nvidia_container_runtime
    create_service_user
    fetch_repo
    pull_model
    write_env_file
    install_systemd_unit
    set_power_mode
    summary
}

main "$@"
