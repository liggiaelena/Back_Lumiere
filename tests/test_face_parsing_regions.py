import sys
from pathlib import Path

import numpy as np


DEV_DIR = Path(__file__).resolve().parents[1] / "dev"
if str(DEV_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DIR))

from app.face_parsing_bisenet import build_face_region_masks, extract_region_crops


def _synthetic_face():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    parsing = np.zeros((100, 100), dtype=np.uint8)
    parsing[10:90, 10:90] = 1
    parsing[38:65, 43:57] = 10
    return image, parsing


def test_cheeks_stay_on_opposite_sides_and_exclude_nose():
    image, parsing = _synthetic_face()
    masks = build_face_region_masks(image, parsing)

    left = masks["bochecha_e"]
    right = masks["bochecha_d"]
    nose = masks["nariz"]

    assert not np.any(left & right)
    assert not np.any(left & nose)
    assert not np.any(right & nose)
    assert np.where(left)[1].max() < np.where(right)[1].min()


def test_display_bbox_does_not_include_crop_padding():
    image, parsing = _synthetic_face()
    masks = build_face_region_masks(image, parsing)
    crops = extract_region_crops(image, masks)
    cheek = crops["bochecha_e"]

    assert cheek["crop_bbox"]["x1"] < cheek["bbox"]["x1"]
    assert cheek["crop_bbox"]["y1"] < cheek["bbox"]["y1"]
