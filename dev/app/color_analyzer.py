"""Stage 4: healthy-skin color extraction outside SegFormer conditions."""

import cv2
import numpy as np

from app.skin_tone_analyzer import analyze_skin_tone


def _to_hex(rgb: list[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def analyze_region_colors(
    img_rgb: np.ndarray,
    parsing_map: np.ndarray,
    condition_mask: np.ndarray,
    region_masks: dict,
) -> dict:
    image = np.asarray(img_rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    parsing = np.asarray(parsing_map, dtype=np.uint8)
    conditions = np.asarray(condition_mask, dtype=np.uint8)
    if parsing.shape != (h, w):
        parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)
    if conditions.shape != (h, w):
        conditions = cv2.resize(conditions, (w, h), interpolation=cv2.INTER_NEAREST)

    valid_skin = (parsing == 1) | (parsing == 10)
    output = {}
    for name, region in region_masks.items():
        region_skin = np.asarray(region, dtype=bool) & valid_skin
        healthy = region_skin & (conditions == 0)
        pixels = image[healthy]
        rgb = np.median(pixels, axis=0).astype(int).tolist() if pixels.size else None
        total = int(region_skin.sum())
        count = int(healthy.sum())
        output[name] = {
            "rgb": rgb,
            "hex": _to_hex(rgb) if rgb is not None else None,
            "healthy_pixel_count": count,
            "excluded_condition_pixel_count": total - count,
            "healthy_percent": round(count / total * 100, 4) if total else 0.0,
        }
    return output


__all__ = ["analyze_skin_tone", "analyze_region_colors"]
