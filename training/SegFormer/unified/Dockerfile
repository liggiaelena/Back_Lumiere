FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Experiment config (NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
# CLASS_WEIGHTS, FEATURE_NAMES, EXPECTED_ACCURACY, EXPERIMENT_NAME,
# EXPERIMENT_VERSION) is intentionally NOT set here. It is injected at
# `docker compose run` time from the root .env file (gitignored), so no
# value is hardcoded in this image or committed to source control. If a
# variable is left unset, train_unified.py falls back to its own default.

WORKDIR /app

COPY requirements.txt .
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

COPY train_unified.py .

# data-collection/ and training/checkpoints/ are bind-mounted at runtime
# (datasets and model checkpoints are too large/volatile to bake into the image).
ENTRYPOINT ["python", "train_unified.py"]
