import cv2
import numpy as np
from PIL import Image, ImageOps
import base64, io
from app.config import settings

def load_and_validate(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not read the image. Please try a different file.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fix EXIF orientation (mobile photos)
    pil = Image.fromarray(img_rgb)
    pil = ImageOps.exif_transpose(pil)
    img_rgb = np.array(pil)

    h, w = img_rgb.shape[:2]
    if min(h, w) < 300:
        raise ValueError("Image too small. Please use at least 300x300px.")

    return img_rgb


def validate_face_brightness(face_img_rgb: np.ndarray) -> None:
    """Check exposure on the detected/cropped face region, not the whole
    photo — a dark background or bright wall behind a well-lit face (or
    vice versa) shouldn't affect this check."""
    mean_brightness = face_img_rgb.mean()
    if mean_brightness < 20:
        raise ValueError("Face region too dark. Please improve the lighting.")
    if mean_brightness > 235:
        raise ValueError("Face region too bright or overexposed.")


def preprocess(img_rgb: np.ndarray) -> dict:
    h, w = img_rgb.shape[:2]
    max_side = settings.max_image_side
    scale = min(max_side / h, max_side / w, 1.0)

    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {"array": img_rgb, "base64": b64, "shape": img_rgb.shape[:2]}
