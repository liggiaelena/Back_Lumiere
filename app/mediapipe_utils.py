import cv2
import numpy as np
import mediapipe as mp
import base64, io
from PIL import Image

FACE_REGIONS = {
    "testa":      [10, 338, 297, 332, 284, 251, 389, 109, 67, 103],
    "bochecha_e": [36, 31, 228, 229, 230, 231, 232, 233, 128, 121],
    "bochecha_d": [266, 261, 448, 449, 450, 451, 452, 453, 357, 350],
    "nariz":      [1, 2, 98, 327, 168, 197, 195, 5],
    "queixo":     [175, 199, 200, 18, 152, 171, 148],
}

def get_landmarks(img_array: np.ndarray) -> dict:
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7
    ) as face_mesh:
        results = face_mesh.process(img_array)

    if not results.multi_face_landmarks:
        raise ValueError("Nenhum rosto detectado. Centralize o rosto na foto e garanta boa iluminação.")

    h, w = img_array.shape[:2]
    lm = results.multi_face_landmarks[0].landmark

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

        pil_crop = Image.fromarray(crop)
        buf = io.BytesIO()
        pil_crop.save(buf, format="JPEG", quality=85)
        b64_crop = base64.b64encode(buf.getvalue()).decode()

        crops[region] = {"array": crop, "base64": b64_crop}

    return crops
