import sys
from pathlib import Path

import numpy as np


DEV_DIR = Path(__file__).resolve().parents[1] / "dev"
if str(DEV_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DIR))

from app.color_analyzer import analyze_region_colors


def test_region_colors_exclude_segformer_condition_pixels():
    image = np.array(
        [
            [[100, 110, 120], [250, 0, 0]],
            [[100, 110, 120], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    parsing = np.ones((2, 2), dtype=np.uint8)
    conditions = np.array([[0, 2], [0, 0]], dtype=np.uint8)
    regions = {"testa": np.ones((2, 2), dtype=bool)}

    result = analyze_region_colors(
        image, parsing, conditions, regions
    )["testa"]

    assert result["rgb"] == [100, 110, 120]
    assert result["hex"] == "#646e78"
    assert result["healthy_pixel_count"] == 3
    assert result["excluded_condition_pixel_count"] == 1
    assert result["healthy_percent"] == 75.0


def test_nose_region_uses_parsed_skin_for_color_analysis():
    image = np.full((2, 2, 3), [80, 90, 100], dtype=np.uint8)
    parsing = np.ones((2, 2), dtype=np.uint8)
    conditions = np.zeros((2, 2), dtype=np.uint8)
    regions = {"nariz": np.ones((2, 2), dtype=bool)}

    result = analyze_region_colors(
        image, parsing, conditions, regions
    )["nariz"]

    assert result["rgb"] == [80, 90, 100]
    assert result["healthy_pixel_count"] == 4
