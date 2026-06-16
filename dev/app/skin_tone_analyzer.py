
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
 
# ── Resolve paths relative to this file ─────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_DIR / "training"
CHECKPOINT = TRAINING_DIR / "checkpoints" / "bisenet_best.pth"
NUM_CLASSES = 14
SKIN_LABEL  = 1       # BiSeNet label index for skin region
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
 
# ── Lazy-load model once at startup, not on every request ────────────────────
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
 
 
def analyze_skin_tone(img_rgb: np.ndarray, bisenet_map: np.ndarray, condition_mask: np.ndarray) -> dict:
    """Analyze overall and per-zone skin tone using a BiSeNet parsing map

    Args:
        img_rgb: HxWx3 RGB image as NumPy array (uint8 recommended).
        bisenet_map: HxW integer parsing map (BiSeNet output).
        condition_mask: HxW binary/integer mask where non-zero denotes
                        abnormal/diseased pixels to exclude.

    Returns: dictionary matching the required JSON schema.
    """
    try:
        img = np.asarray(img_rgb)
        h, w = img.shape[:2]

        # Ensure masks are same spatial size
        if bisenet_map.shape != (h, w):
            raise ValueError("bisenet_map shape must match img_rgb height and width")
        if condition_mask.shape != (h, w):
            raise ValueError("condition_mask shape must match img_rgb height and width")

        # Compute face bbox where parsing > 0
        face_mask = bisenet_map > 0
        if not face_mask.any():
            # No face detected — return empty structured result
            return {
                "median_hex": None,
                "median_rgb": [0, 0, 0],
                "num_skin_pixels": 0,
                "zones": {
                    "forehead":      {"hex": None, "rgb": [0, 0, 0]},
                    "under_eyes":    {"hex": None, "rgb": [0, 0, 0]},
                    "cheeks":        {"hex": None, "rgb": [0, 0, 0]},
                    "around_mouth":  {"hex": None, "rgb": [0, 0, 0]},
                    "jawline":       {"hex": None, "rgb": [0, 0, 0]},
                }
            }

        ys, xs = np.where(face_mask)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        f_h = y_max - y_min + 1
        f_w = x_max - x_min + 1

        # Healthy skin mask: strictly (skin label) & (condition_mask == 0)
        healthy_mask = (bisenet_map == SKIN_LABEL) & (condition_mask == 0)

        # Fallback global skin median (from all skin-labelled pixels regardless of condition)
        all_skin_mask = (bisenet_map == SKIN_LABEL)
        all_skin_pixels = img[all_skin_mask]
        if all_skin_pixels.size > 0:
            global_median = np.median(all_skin_pixels, axis=0).astype(int).tolist()
        else:
            global_median = [0, 0, 0]

        # Primary median computed from healthy skin; fallback to global_median
        healthy_pixels = img[healthy_mask]
        if healthy_pixels.size > 0:
            primary_median = np.median(healthy_pixels, axis=0).astype(int).tolist()
        else:
            primary_median = global_median

        # Zone masks
        zone_masks = {}

        # Forehead: y < y_min + 0.3 * f_h
        y_forehead_thresh = y_min + int(0.3 * f_h)
        mask_forehead = healthy_mask.copy()
        yy = np.arange(h)[:, None]
        mask_forehead &= (yy < y_forehead_thresh)
        zone_masks["forehead"] = mask_forehead

        # Under eyes: localized below eyes (labels 4 and 5)
        eye_mask = (bisenet_map == 4) | (bisenet_map == 5)
        mask_under_eyes = np.zeros_like(healthy_mask, dtype=bool)
        if eye_mask.any():
            eye_ys, eye_xs = np.where(eye_mask)
            eye_ymax = int(eye_ys.max())
            # region from eye_ymax to eye_ymax + 0.12 * face_height
            y_start = eye_ymax
            y_end = min(h - 1, eye_ymax + int(0.12 * f_h) + 1)
            # horizontal extent: bounding box of eyes expanded by 10% face width
            ex_min = int(max(0, eye_xs.min() - 0.1 * f_w))
            ex_max = int(min(w - 1, eye_xs.max() + 0.1 * f_w))
            yy = np.arange(h)[:, None]
            xx = np.arange(w)[None, :]
            mask_under_eyes = (yy >= y_start) & (yy <= y_end) & (xx >= ex_min) & (xx <= ex_max)
            mask_under_eyes &= healthy_mask
        zone_masks["under_eyes"] = mask_under_eyes

        # Cheeks: left and right thirds, bounded vertically to avoid forehead/jaw
        x_left_thresh = x_min + int(0.33 * f_w)
        x_right_thresh = x_min + int(0.66 * f_w)
        y_vert_min = y_min + int(0.25 * f_h)
        y_vert_max = y_min + int(0.75 * f_h)
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        mask_cheeks = ((xx < x_left_thresh) | (xx > x_right_thresh)) & (yy >= y_vert_min) & (yy <= y_vert_max)
        mask_cheeks &= healthy_mask
        zone_masks["cheeks"] = mask_cheeks

        # Around mouth: dilated lips (labels 12 and 13) excluding lips themselves
        lips_mask = (bisenet_map == 12) | (bisenet_map == 13)
        mask_around_mouth = np.zeros_like(healthy_mask, dtype=bool)
        if lips_mask.any():
            # dilate lips region
            lip_uint = lips_mask.astype(np.uint8)
            k = max(3, int(0.03 * max(h, w)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 6 + 1, k * 3 + 1))
            dilated = cv2.dilate(lip_uint, kernel)
            mask_around_mouth = (dilated.astype(bool)) & (~lips_mask)
            mask_around_mouth &= healthy_mask
        zone_masks["around_mouth"] = mask_around_mouth

        # Jawline: y > y_min + 0.8 * f_h
        y_jaw_thresh = y_min + int(0.8 * f_h)
        yy = np.arange(h)[:, None]   #  (1024, 1)
        #mask_jawline = (yy > y_jaw_thresh)  # (1024, 1)
        mask_jawline = (yy > y_jaw_thresh) & healthy_mask
        zone_masks["jawline"] = mask_jawline

        # Compute medians per zone with fallback to primary_median then global_median
        def _zone_median(mask, fallback_rgb):
            pixels = img[mask]
            if pixels.size == 0:
                return fallback_rgb
            return np.median(pixels, axis=0).astype(int).tolist()

        zones_out = {}
        for name in ["forehead", "under_eyes", "cheeks", "around_mouth", "jawline"]:
            rgb = _zone_median(zone_masks[name], primary_median)
            # if still empty (primary == global) rgb will be global_median or zeros
            zones_out[name] = {
                "hex": _to_hex(rgb) if rgb is not None else None,
                "rgb": rgb
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
        return {
            "median_hex": None,
            "median_rgb": [0, 0, 0],
            "num_skin_pixels": 0,
            "zones": {
                "forehead":      {"hex": None, "rgb": [0, 0, 0]},
                "under_eyes":    {"hex": None, "rgb": [0, 0, 0]},
                "cheeks":        {"hex": None, "rgb": [0, 0, 0]},
                "around_mouth":  {"hex": None, "rgb": [0, 0, 0]},
                "jawline":       {"hex": None, "rgb": [0, 0, 0]},
            }
        }
 
 
# ── Helpers
def _to_hex(rgb: list) -> str:
    if rgb is None:
        return None
    # Ensure ints and clamp
    r, g, b = [int(max(0, min(255, int(x)))) for x in rgb]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)
 
# def _empty_result() -> dict:
#     return {
#         "mean_rgb":        None,
#         "mean_hex":        None,
#         "median_rgb":      None,
#         "median_hex":      None,
#         "num_skin_pixels": 0,
#     }
