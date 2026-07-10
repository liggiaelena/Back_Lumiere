FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY dev/app/ ./dev/app/
COPY dev/run.py ./dev/run.py

COPY training/models/ ./training/models/

RUN mkdir -p /app/training/checkpoints/SegFormer /app/training/checkpoints

CMD ["python", "dev/run.py"]