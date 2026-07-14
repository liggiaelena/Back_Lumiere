import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dev"))

from app.color_utils import hex_to_fitzpatrick


class HexToFitzpatrickTests(unittest.TestCase):
    def test_light_pink_skin_is_not_classified_as_type_v(self):
        self.assertLessEqual(hex_to_fitzpatrick("#e7b49b"), 3)
        self.assertLessEqual(hex_to_fitzpatrick("#f0c8b0"), 3)

    def test_dark_reference_still_maps_to_type_v(self):
        self.assertEqual(hex_to_fitzpatrick("#7d4e2d"), 5)

    def test_reference_scale_remains_ordered(self):
        samples = ["#f6ede4", "#f3d7c0", "#dba98a", "#b07d5b", "#7d4e2d", "#3e1f0e"]
        values = [hex_to_fitzpatrick(sample) for sample in samples]

        self.assertEqual(values, sorted(values))


if __name__ == "__main__":
    unittest.main()
