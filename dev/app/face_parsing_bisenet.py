"""Stage 3: 19-class BiSeNet face parsing and five-region extraction."""

import base64
import io
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_DIR / "training"
CHECKPOINT = TRAINING_DIR / "checkpoints" / "BiSeNet" / "79999_iter.pth"
NUM_CLASSES = 19
SKIN_LABEL = 1
NOSE_LABEL = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

_model = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not CHECKPOINT.exists():
                    raise FileNotFoundError(f"BiSeNet checkpoint is missing: {CHECKPOINT}")
                from models.model import BiSeNet

                model = BiSeNet(n_classes=NUM_CLASSES)
                state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state)
                model.eval().to(DEVICE)
                _model = model
    return _model


def parse_face(img_rgb: np.ndarray) -> np.ndarray:
    """Return a 19-class parsing map at the input image resolution."""
    image = np.asarray(img_rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])(Image.fromarray(image).convert("RGB")).unsqueeze(0).to(DEVICE)

    with _inference_lock, torch.no_grad():
        logits = _get_model()(tensor)[0]
        parsing = logits.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)


def build_face_region_masks(img_rgb: np.ndarray, parsing_map: np.ndarray) -> dict:
    """Build forehead, left/right cheek, nose and chin masks."""
    h, w = np.asarray(img_rgb).shape[:2]
    parsing = np.asarray(parsing_map, dtype=np.uint8)
    if parsing.shape != (h, w):
        parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)

    skin = parsing == SKIN_LABEL
    if not skin.any():
        raise ValueError("BiSeNet did not detect a usable face/skin region.")

    # Use facial skin, not every non-background class. The latter also contains
    # hair, neck and clothes and makes the geometric face box far too large.
    ys, xs = np.where(skin)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    fw, fh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]

    nose = parsing == NOSE_LABEL
    if not nose.any():
        nose = (
            skin
            & (xx >= x1 + int(0.36 * fw))
            & (xx <= x1 + int(0.64 * fw))
            & (yy >= y1 + int(0.30 * fh))
            & (yy <= y1 + int(0.72 * fh))
        )
    cheek_band = skin & (yy >= y1 + int(0.34 * fh)) & (yy <= y1 + int(0.74 * fh))
    masks = {
        "testa": skin & (yy < y1 + int(0.32 * fh)),
        # Leave a central gap around the nose/mouth instead of splitting the
        # whole skin mask at the center line. Names follow screen orientation.
        "bochecha_e": (
            cheek_band
            & (xx >= x1 + int(0.06 * fw))
            & (xx <= x1 + int(0.43 * fw))
            & ~nose
        ),
        "bochecha_d": (
            cheek_band
            & (xx >= x1 + int(0.57 * fw))
            & (xx <= x1 + int(0.94 * fw))
            & ~nose
        ),
        "nariz": nose,
        "queixo": skin & (yy > y1 + int(0.72 * fh)),
    }
    if any(not mask.any() for mask in masks.values()):
        missing = [name for name, mask in masks.items() if not mask.any()]
        raise ValueError(f"BiSeNet face regions are incomplete: {', '.join(missing)}")
    return masks


def extract_region_crops(img_rgb: np.ndarray, region_masks: dict) -> dict:
    """Return masked JPEG crops plus pixel and percentage positions."""
    image = np.asarray(img_rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    output = {}
    for name, mask in region_masks.items():
        ys, xs = np.where(mask)
        region_x1, region_y1 = int(xs.min()), int(ys.min())
        region_x2, region_y2 = int(xs.max()) + 1, int(ys.max()) + 1
        crop_x1, crop_y1 = max(0, region_x1 - 15), max(0, region_y1 - 15)
        crop_x2, crop_y2 = min(w, region_x2 + 15), min(h, region_y2 + 15)
        masked = np.zeros_like(image)
        masked[mask] = image[mask]
        crop = masked[crop_y1:crop_y2, crop_x1:crop_x2]
        buffer = io.BytesIO()
        Image.fromarray(crop).save(buffer, format="JPEG", quality=85)
        output[name] = {
            "array": crop,
            "base64": base64.b64encode(buffer.getvalue()).decode(),
            "bbox": {
                "x1": region_x1,
                "y1": region_y1,
                "x2": region_x2,
                "y2": region_y2,
            },
            "crop_bbox": {
                "x1": crop_x1,
                "y1": crop_y1,
                "x2": crop_x2,
                "y2": crop_y2,
            },
            "bbox_percent": {
                "x": round(region_x1 / w * 100, 4),
                "y": round(region_y1 / h * 100, 4),
                "width": round((region_x2 - region_x1) / w * 100, 4),
                "height": round((region_y2 - region_y1) / h * 100, 4),
            },
        }
    return output


__all__ = ["parse_face", "build_face_region_masks", "extract_region_crops"]
