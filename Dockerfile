FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgl1/libglib2.0-0/libgles2/libegl1: required by mediapipe's native
# runtime (it dlopens GLES/EGL even for CPU-only, no-display inference).
# curl: used below to bake the face_landmarker model into the image at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgles2 \
    libegl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch/torchvision first — the default PyPI wheels pull in
# CUDA libraries and add several GB that this project never uses.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Pre-download the MediaPipe face landmarker so the container never needs
# outbound internet access on first request (mediapipe_utils.py downloads
# this to ~/face_landmarker.task at runtime if it isn't already there).
RUN curl -L -o /root/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Pre-download the ImageNet-pretrained ResNet18 backbone that BiSeNet's
# Resnet18.init_weight() fetches unconditionally on construction (training/models/resnet.py).
# Its weights are immediately overwritten by bisenet_best.pth, but the download
# still happens on every fresh container unless it's already in torch's cache dir.
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    curl -L -o /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth \
    https://download.pytorch.org/models/resnet18-5c106cde.pth

COPY dev/ dev/
COPY training/models training/models
COPY training/checkpoints/bisenet_best.pth training/checkpoints/bisenet_best.pth
COPY training/checkpoints/SegFormer/unified/best training/checkpoints/SegFormer/unified/best

WORKDIR /app/dev

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
