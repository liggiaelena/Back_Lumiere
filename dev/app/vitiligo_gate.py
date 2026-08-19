"""Image-level vitiligo classifier used to suppress segmentation false positives."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    PROJECT_DIR / "training" / "checkpoints" / "VitiligoClassifier" / "resnet18"
)
_BUNDLE: Optional[dict] = None
_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _model_dir() -> Path:
    configured = os.getenv("VITILIGO_CLASSIFIER_DIR")
    return Path(configured) if configured else DEFAULT_MODEL_DIR


def _get_bundle() -> Optional[dict]:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    directory = _model_dir()
    checkpoint_path = directory / "best.pt"
    deployment_path = directory / "deployment.json"
    if not checkpoint_path.exists() or not deployment_path.exists():
        return None
    with _LOAD_LOCK:
        if _BUNDLE is None:
            deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(model.fc.in_features, 2))
            state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state["model"])
            model.to(DEVICE).eval()
            _BUNDLE = {
                "model": model,
                "threshold": float(deployment["vitiligo_threshold"]),
                "path": str(checkpoint_path),
            }
    return _BUNDLE


def predict_vitiligo_probability(img_rgb: np.ndarray) -> Optional[float]:
    bundle = _get_bundle()
    if bundle is None:
        return None
    image = np.asarray(img_rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = 256.0 / min(h, w)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    y = (resized.shape[0] - 224) // 2
    x = (resized.shape[1] - 224) // 2
    crop = resized[y:y + 224, x:x + 224]
    tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = ((tensor - mean) / std).unsqueeze(0).to(DEVICE)
    with _INFERENCE_LOCK, torch.inference_mode():
        probability = bundle["model"](tensor).softmax(1)[0, 1]
    return float(probability.cpu())


def apply_vitiligo_gate(img_rgb: np.ndarray, condition_mask: np.ndarray, condition_map: dict) -> None:
    probability = predict_vitiligo_probability(img_rgb)
    if probability is None:
        condition_map["vitiligo"]["classifier_gate"] = {"available": False}
        return
    bundle = _get_bundle()
    threshold = bundle["threshold"]
    passed = probability >= threshold
    condition_map["vitiligo"]["classifier_gate"] = {
        "available": True,
        "probability": round(probability, 6),
        "threshold": round(threshold, 6),
        "passed": passed,
    }
    if condition_map["vitiligo"].get("detected") and not passed:
        condition_mask[condition_mask == 1] = 0
        condition_map["vitiligo"].update({
            "detected": False, "area_percent": 0, "zones": [],
            "suppressed_by_classifier": True,
        })
