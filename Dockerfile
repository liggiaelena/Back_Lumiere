FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libgles2 libegl1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# 預先下載模型
RUN mkdir -p /root/ && curl -L -o /root/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
RUN mkdir -p /root/.cache/torch/hub/checkpoints && curl -L -o /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth https://download.pytorch.org/models/resnet18-5c106cde.pth

CMD ["uvicorn", "dev.app.main:app", "--host", "0.0.0.0", "--port", "8001"]