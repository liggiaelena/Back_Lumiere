import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dev"))

from app.recommendations import _match_by_color


class DatabaseRecommendationTests(unittest.TestCase):
    def test_returns_product_metadata_and_click_url(self):
        database = [{
            "product_id": 7, "brand": "Example", "product_name": "Foundation",
            "product_url": "https://example.test/product", "image_url": None,
            "currency": "USD", "price": 25, "ingredients": ["Water"],
            "allergens": [], "fetched_at": "2026-08-07T00:00:00Z",
            "shade_name": "N1", "shade_code": "N1", "shade_hex": "#dba98a",
            "undertone": "neutro",
        }]
        matches, reliable = _match_by_color("#dba98a", ["neutro", "quente", "frio"], 1, database)
        self.assertTrue(reliable)
        self.assertEqual(matches[0]["product_url"], "https://example.test/product")
        self.assertEqual(matches[0]["price"], 25.0)

    def test_empty_filtered_catalog_returns_no_matches(self):
        matches, reliable = _match_by_color("#dba98a", ["neutro"], 1, [])
        self.assertEqual(matches, [])
        self.assertFalse(reliable)


if __name__ == "__main__":
    unittest.main()
