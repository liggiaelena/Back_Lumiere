import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _find_project_dir() -> Path:
    current = Path(__file__).resolve()

    for parent in [current.parent] + list(current.parents):
        if (parent / "training" / "checkpoints").exists():
            return parent

    try:
        return current.parents[2]
    except IndexError:
        return current.parent


PROJECT_DIR = _find_project_dir()
TRAINING_DIR = PROJECT_DIR / "training"
CHECKPOINT = TRAINING_DIR / "checkpoints" / "BiSeNet" / "bisenet_best.pth"

NUM_CLASSES = 14
SKIN_LABEL = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


_model = None


def _get_model():
    global _model

    if _model is None:
        from models.model import BiSeNet

        net = BiSeNet(n_classes=NUM_CLASSES)
        state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        net.load_state_dict(state)
        net.eval()
        net.to(DEVICE)

        _model = net

    return _model


def analyze_skin_tone(
    img_rgb: np.ndarray,
    bisenet_map: np.ndarray,
    condition_mask: np.ndarray = None,
) -> dict:
    try:
        img = np.asarray(img_rgb)
        h, w = img.shape[:2]

        if bisenet_map.shape != (h, w):
            bisenet_map = cv2.resize(
                bisenet_map.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        if condition_mask is None:
            condition_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            condition_mask = np.asarray(condition_mask)

            if condition_mask.shape != (h, w):
                condition_mask = cv2.resize(
                    condition_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

        face_mask = bisenet_map > 0

        if not face_mask.any():
            return _empty_result()

        ys, xs = np.where(face_mask)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        f_h = y_max - y_min + 1
        f_w = x_max - x_min + 1

        healthy_mask = (bisenet_map == SKIN_LABEL) & (condition_mask == 0)

        all_skin_mask = bisenet_map == SKIN_LABEL
        all_skin_pixels = img[all_skin_mask]

        if all_skin_pixels.size > 0:
            global_median = np.median(all_skin_pixels, axis=0).astype(int).tolist()
        else:
            global_median = [0, 0, 0]

        healthy_pixels = img[healthy_mask]

        if healthy_pixels.size > 0:
            primary_median = np.median(healthy_pixels, axis=0).astype(int).tolist()
        else:
            primary_median = global_median

        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]

        zone_masks = {}

        y_forehead_thresh = y_min + int(0.30 * f_h)
        zone_masks["forehead"] = healthy_mask & (yy < y_forehead_thresh)

        eye_mask = (bisenet_map == 4) | (bisenet_map == 5)
        mask_under_eyes = np.zeros_like(healthy_mask, dtype=bool)

        if eye_mask.any():
            eye_ys, eye_xs = np.where(eye_mask)
            eye_ymax = int(eye_ys.max())

            y_start = eye_ymax
            y_end = min(h - 1, eye_ymax + int(0.12 * f_h) + 1)

            ex_min = int(max(0, eye_xs.min() - 0.10 * f_w))
            ex_max = int(min(w - 1, eye_xs.max() + 0.10 * f_w))

            mask_under_eyes = (
                (yy >= y_start)
                & (yy <= y_end)
                & (xx >= ex_min)
                & (xx <= ex_max)
                & healthy_mask
            )

        zone_masks["under_eyes"] = mask_under_eyes

        x_left_thresh = x_min + int(0.33 * f_w)
        x_right_thresh = x_min + int(0.66 * f_w)
        y_vert_min = y_min + int(0.25 * f_h)
        y_vert_max = y_min + int(0.75 * f_h)

        zone_masks["cheeks"] = (
            ((xx < x_left_thresh) | (xx > x_right_thresh))
            & (yy >= y_vert_min)
            & (yy <= y_vert_max)
            & healthy_mask
        )

        lips_mask = (bisenet_map == 12) | (bisenet_map == 13)
        mask_around_mouth = np.zeros_like(healthy_mask, dtype=bool)

        if lips_mask.any():
            lip_uint = lips_mask.astype(np.uint8)

            k = max(3, int(0.03 * max(h, w)))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (k * 6 + 1, k * 3 + 1),
            )

            dilated = cv2.dilate(lip_uint, kernel)
            mask_around_mouth = dilated.astype(bool) & (~lips_mask) & healthy_mask

        zone_masks["around_mouth"] = mask_around_mouth

        y_jaw_thresh = y_min + int(0.80 * f_h)
        zone_masks["jawline"] = (yy > y_jaw_thresh) & healthy_mask

        def _zone_median(mask, fallback_rgb):
            pixels = img[mask]

            if pixels.size == 0:
                return fallback_rgb

            return np.median(pixels, axis=0).astype(int).tolist()

        zones_out = {}

        for name in ["forehead", "under_eyes", "cheeks", "around_mouth", "jawline"]:
            rgb = _zone_median(zone_masks[name], primary_median)

            zones_out[name] = {
                "hex": _to_hex(rgb) if rgb is not None else None,
                "rgb": rgb,
            }

        median_rgb = primary_median if primary_median is not None else global_median

        return {
            "median_hex": _to_hex(median_rgb) if median_rgb is not None else None,
            "median_rgb": median_rgb,
            "num_skin_pixels": int(healthy_mask.sum()),
            "zones": zones_out,
        }

    except Exception:
        import traceback

        traceback.print_exc()
        return _empty_result()


def analyze_region_skin_colors(
    img_rgb: np.ndarray,
    bisenet_map: np.ndarray,
    condition_mask: np.ndarray,
    region_masks: dict,
) -> dict:
    """Extract median healthy-skin color for each parsed facial region.

    A usable pixel must belong to BiSeNet's skin class, belong to the requested
    face region, and not belong to any SegFormer condition class.
    """
    image = np.asarray(img_rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    parsing = np.asarray(bisenet_map, dtype=np.uint8)
    conditions = np.asarray(condition_mask, dtype=np.uint8)
    if parsing.shape != (h, w):
        parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)
    if conditions.shape != (h, w):
        conditions = cv2.resize(
            conditions, (w, h), interpolation=cv2.INTER_NEAREST
        )

    # Only the checkpoint's facial-skin class is eligible. Semantic parts such
    # as eyes, lips, hair and uncertain class-10 predictions stay excluded.
    skin_mask = parsing == SKIN_LABEL
    healthy_skin_mask = skin_mask & (conditions == 0)
    output = {}
    for name, raw_region_mask in region_masks.items():
        region_mask = np.asarray(raw_region_mask, dtype=bool)
        if region_mask.shape != (h, w):
            region_mask = cv2.resize(
                region_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        region_skin = skin_mask & region_mask
        healthy_region = healthy_skin_mask & region_mask
        pixels = image[healthy_region]
        rgb = (
            np.median(pixels, axis=0).astype(int).tolist()
            if pixels.size
            else None
        )
        total = int(region_skin.sum())
        healthy = int(healthy_region.sum())
        output[name] = {
            "rgb": rgb,
            "hex": _to_hex(rgb) if rgb is not None else None,
            "healthy_pixel_count": healthy,
            "excluded_condition_pixel_count": total - healthy,
            "healthy_percent": round(healthy / total * 100, 4) if total else 0.0,
        }
    return output


def _to_hex(rgb: list) -> str:
    if rgb is None:
        return None

    r, g, b = [int(max(0, min(255, int(x)))) for x in rgb]

    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def _empty_result() -> dict:
    return {
        "median_hex": None,
        "median_rgb": [0, 0, 0],
        "num_skin_pixels": 0,
        "zones": {
            "forehead": {"hex": None, "rgb": [0, 0, 0]},
            "under_eyes": {"hex": None, "rgb": [0, 0, 0]},
            "cheeks": {"hex": None, "rgb": [0, 0, 0]},
            "around_mouth": {"hex": None, "rgb": [0, 0, 0]},
            "jawline": {"hex": None, "rgb": [0, 0, 0]},
        },
    }
