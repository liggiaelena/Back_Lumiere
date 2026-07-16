"""MediaPipe face detection and face-focused image preparation."""

import threading
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "blaze_face_short_range.tflite"

_detector = None
_detector_lock = threading.Lock()
_inference_lock = threading.Lock()


def _get_detector():
    """Create one shared detector; the model is deployed with the project."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"MediaPipe face detector model is missing: {MODEL_PATH}"
                    )
                options = vision.FaceDetectorOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(MODEL_PATH),
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                    min_suppression_threshold=0.3,
                )
                _detector = vision.FaceDetector.create_from_options(options)
    return _detector


def _pick_largest_detection(detections):
    if not detections:
        return None
    return max(
        detections,
        key=lambda detection: (
            detection.bounding_box.width * detection.bounding_box.height
        ),
    )


def detect_and_zoom_face(
    img_rgb: np.ndarray,
    padding_ratio: float = 0.35,
) -> tuple[np.ndarray, dict]:
    """Find the largest face and crop a wider, face-focused RGB image.

    Cropping to an expanded face box creates the zoomed view without resizing
    pixels, which avoids introducing interpolation artefacts into skin analysis.
    """
    image = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Face detection expects an RGB image with three channels.")
    if not 0 <= padding_ratio <= 1:
        raise ValueError("padding_ratio must be between 0 and 1.")

    h, w = image.shape[:2]
    with _inference_lock:
        result = _get_detector().detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        )

    best = _pick_largest_detection(result.detections)
    if best is None:
        raise ValueError(
            "No face detected. Please center your face and use adequate lighting."
        )

    box = best.bounding_box
    face_x1 = max(0, int(box.origin_x))
    face_y1 = max(0, int(box.origin_y))
    face_x2 = min(w, int(box.origin_x + box.width))
    face_y2 = min(h, int(box.origin_y + box.height))
    if face_x2 <= face_x1 or face_y2 <= face_y1:
        raise ValueError("The detected face bounding box is invalid.")

    # BlazeFace boxes are tight. Add room for forehead, chin, ears and context.
    pad_x = int((face_x2 - face_x1) * padding_ratio)
    pad_y = int((face_y2 - face_y1) * padding_ratio)
    crop_x1 = max(0, face_x1 - pad_x)
    crop_y1 = max(0, face_y1 - pad_y)
    crop_x2 = min(w, face_x2 + pad_x)
    crop_y2 = min(h, face_y2 + pad_y)
    zoomed = np.ascontiguousarray(image[crop_y1:crop_y2, crop_x1:crop_x2])

    keypoints = [
        {
            "x": int(point.x * w),
            "y": int(point.y * h),
            "x_percent": round(float(point.x * 100), 4),
            "y_percent": round(float(point.y * 100), 4),
        }
        for point in best.keypoints
    ]
    info = {
        "bbox": {
            "x1": face_x1,
            "y1": face_y1,
            "x2": face_x2,
            "y2": face_y2,
        },
        "bbox_percent": {
            "x": round(face_x1 / w * 100, 4),
            "y": round(face_y1 / h * 100, 4),
            "width": round((face_x2 - face_x1) / w * 100, 4),
            "height": round((face_y2 - face_y1) / h * 100, 4),
        },
        "crop_bbox": {
            "x1": crop_x1,
            "y1": crop_y1,
            "x2": crop_x2,
            "y2": crop_y2,
        },
        "image_size": {"width": w, "height": h},
        "zoomed_image_size": {
            "width": int(zoomed.shape[1]),
            "height": int(zoomed.shape[0]),
        },
        "padding_ratio": padding_ratio,
        "confidence": round(float(best.categories[0].score), 6),
        "keypoints": keypoints,
    }
    return zoomed, info


def detect_face(img_rgb: np.ndarray) -> dict:
    """Backward-compatible detection API."""
    _, info = detect_and_zoom_face(img_rgb)
    return info
