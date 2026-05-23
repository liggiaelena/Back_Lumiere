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
