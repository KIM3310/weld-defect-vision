# Jetson Orin (JetPack 6.1, L4T r36.4) inference container for weld-defect-vision.
#
# Base image includes CUDA 12.6, cuDNN 9.3, TensorRT 10.3, and the PyTorch build
# NVIDIA ships for Orin. We layer the project requirements on top; Triton runs
# as a separate container (see serving/triton/docker-compose.yml) and this
# container only runs the Python capture daemon that drives it.
#
# Build on the Orin itself (the base image is multi-arch aarch64/linux):
#   docker build -t weld-defect-vision:orin -f edge/jetson-orin/Dockerfile .

ARG L4T_VERSION=r36.4.0
FROM nvcr.io/nvidia/l4t-pytorch:${L4T_VERSION}-pth2.4-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps. libgl1 is needed for opencv-python-headless is NOT strictly
# needed but libglib2.0-0 is required by cv2 imports even on headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        ca-certificates \
        curl \
        netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy lean Python requirements for the edge container; skip training-only deps.
# The full requirements.txt pulls pandas/seaborn/matplotlib which are unused at
# the edge. We keep a leaner set here and rely on system PyTorch from the base.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        ultralytics \
        opencv-python-headless \
        pillow \
        fastapi \
        uvicorn \
        python-multipart \
        pydantic \
        asyncua \
        paho-mqtt \
        httpx \
        tritonclient[grpc] \
        prometheus-client \
    && python -c "import ultralytics; print(ultralytics.__version__)"

COPY src /app/src
COPY api /app/api
COPY integrations /app/integrations
COPY edge/common /app/edge/common

ENV PYTHONPATH=/app \
    WELD_DEFECT_MODEL_REGISTRY=/opt/weld-defect-vision/models \
    TRITON_URL=triton:8001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default command runs the capture daemon. The FastAPI dev server is available
# as an alternative for manual testing.
CMD ["python", "-m", "integrations.opc_ua.client"]
