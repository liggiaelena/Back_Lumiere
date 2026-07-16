import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import base64, io, os, urllib.request
from PIL import Image

_MODEL_PATH = os.path.join(os.path.expanduser("~"), "face_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("Downloading face_landmarker model (~1 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Model downloaded.")

FACE_REGIONS = {
    "testa":      [10, 338, 297, 332, 284, 251, 389, 109, 67, 103],
    "bochecha_e": [36, 31, 228, 229, 230, 231, 232, 233, 128, 121],
    "bochecha_d": [266, 261, 448, 449, 450, 451, 452, 453, 357, 350],
    "nariz":      [1, 2, 98, 327, 168, 197, 195, 5],
    "queixo":     [175, 199, 200, 18, 152, 32, 262],
}

def get_landmarks(img_array: np.ndarray) -> dict:
    _ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected. Please center your face in the photo and ensure good lighting.")

    h, w = img_array.shape[:2]
    lm = result.face_landmarks[0]

    coords = {}
    for region, indices in FACE_REGIONS.items():
        coords[region] = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]

    return coords


def extract_region_crops(img_array: np.ndarray, coords: dict) -> dict:
    crops = {}
    h, w = img_array.shape[:2]

    for region, points in coords.items():
        pts = np.array(points, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        padding = 15
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 255)

        masked = cv2.bitwise_and(img_array, img_array, mask=mask)
        crop = masked[y1:y2, x1:x2]

        # Filter out very dark pixels (shadow, hair, background) before
        # encoding — keeps the crop representative of actual skin tone.
        # Pixels with all channels below 30 are considered non-skin.
        brightness = crop.max(axis=2)  # per-pixel max channel value
        skin_mask_crop = brightness > 30
        if skin_mask_crop.any():
            clean = crop.copy()
            clean[~skin_mask_crop] = [0, 0, 0]
        else:
            clean = crop  # fallback: keep as-is if everything is dark

        pil_crop = Image.fromarray(clean)
        buf = io.BytesIO()
        pil_crop.save(buf, format="JPEG", quality=85)
        b64_crop = base64.b64encode(buf.getvalue()).decode()

        crops[region] = {"array": clean, "base64": b64_crop}

    return crops


def build_parsing_region_masks(
    img_array: np.ndarray,
    parsing_map: np.ndarray,
) -> dict:
    """Return five named boolean masks derived from a 14-class BiSeNet map.

    The public region keys intentionally stay compatible with the previous
    MediaPipe implementation.  Class 1 is facial skin and class 10 is nose.
    The remaining skin is split relative to the parsed face bounding box. All
    masks use the same coordinate system and dimensions as ``img_array``.
    """
    image = np.asarray(img_array)
    h, w = image.shape[:2]
    parsing = np.asarray(parsing_map, dtype=np.uint8)
    if parsing.shape != (h, w):
        parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)

    face_mask = parsing > 0
    skin_mask = parsing == 1
    if not face_mask.any() or not skin_mask.any():
        raise ValueError("BiSeNet did not detect a usable face/skin region.")

    ys, xs = np.where(face_mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    face_w = max(1, x_max - x_min + 1)
    face_h = max(1, y_max - y_min + 1)
    x_mid = x_min + face_w // 2

    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    forehead = skin_mask & (yy < y_min + int(0.32 * face_h))
    chin = skin_mask & (yy > y_min + int(0.72 * face_h))
    # Keep the nose inside the parsed facial-skin class. Although class 10 is
    # commonly named "nose" in public face-parsing datasets, this project's
    # 14-class checkpoint does not predict it reliably enough to use directly.
    nose = (
        skin_mask
        & (xx >= x_min + int(0.36 * face_w))
        & (xx <= x_min + int(0.64 * face_w))
        & (yy >= y_min + int(0.30 * face_h))
        & (yy <= y_min + int(0.72 * face_h))
    )
    cheek_band = (
        skin_mask
        & (yy >= y_min + int(0.28 * face_h))
        & (yy <= y_min + int(0.76 * face_h))
        & ~nose
    )

    region_masks = {
        "testa": forehead,
        "bochecha_e": cheek_band & (xx < x_mid),
        "bochecha_d": cheek_band & (xx >= x_mid),
        "nariz": nose,
        "queixo": chin,
    }

    # A class may be absent in a difficult image (for example the nose class
    # behind glasses or the bottom skin class in a tight crop). Keep all five
    # API parts available by falling back inside the model-derived face box.
    fallback_boxes = {
        "testa": (0.20, 0.05, 0.80, 0.30),
        "bochecha_e": (0.04, 0.38, 0.46, 0.72),
        "bochecha_d": (0.54, 0.38, 0.96, 0.72),
        "nariz": (0.36, 0.32, 0.64, 0.70),
        "queixo": (0.25, 0.72, 0.75, 0.96),
    }
    for region, region_mask in region_masks.items():
        if region_mask.any():
            continue
        bx1, by1, bx2, by2 = fallback_boxes[region]
        fallback = (
            (xx >= x_min + int(bx1 * face_w))
            & (xx <= x_min + int(bx2 * face_w))
            & (yy >= y_min + int(by1 * face_h))
            & (yy <= y_min + int(by2 * face_h))
        )
        model_fallback = fallback & face_mask
        region_masks[region] = model_fallback if model_fallback.any() else fallback

    return region_masks


def extract_parsing_region_crops(
    img_array: np.ndarray,
    parsing_map: np.ndarray,
    region_masks: dict | None = None,
) -> dict:
    """Crop the five face regions and include their image-space positions."""
    image = np.asarray(img_array)
    h, w = image.shape[:2]
    if region_masks is None:
        region_masks = build_parsing_region_masks(image, parsing_map)

    crops = {}
    padding = 15
    for region, region_mask in region_masks.items():
        region_ys, region_xs = np.where(region_mask)
        rx1 = max(0, int(region_xs.min()) - padding)
        ry1 = max(0, int(region_ys.min()) - padding)
        rx2 = min(w, int(region_xs.max()) + padding + 1)
        ry2 = min(h, int(region_ys.max()) + padding + 1)

        masked = np.zeros_like(image)
        masked[region_mask] = image[region_mask]
        clean = masked[ry1:ry2, rx1:rx2]

        pil_crop = Image.fromarray(clean)
        buf = io.BytesIO()
        pil_crop.save(buf, format="JPEG", quality=85)
        crops[region] = {
            "array": clean,
            "base64": base64.b64encode(buf.getvalue()).decode(),
            "bbox": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
            "bbox_percent": {
                "x": round(rx1 / w * 100, 4),
                "y": round(ry1 / h * 100, 4),
                "width": round((rx2 - rx1) / w * 100, 4),
                "height": round((ry2 - ry1) / h * 100, 4),
            },
        }

    return crops
