import json
import io
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dev"))

from PIL import Image
from product_sync import collect, _discovered_shades, _estimate_swatch_hex


class ProductSyncTests(unittest.TestCase):
    def test_extracts_nested_shop_shades(self):
        document = '''
        <script type="application/json">
        {"product":{"variants":[
          {"shadeName":"Warm Sand", "hexCode":"#D7A27D", "sku":"W4", "available":true},
          {"shadeName":"Cool Rose", "hexCode":"A87565", "sku":"C6", "available":false}
        ]}}
        </script>
        '''
        shades = _discovered_shades(document)
        self.assertEqual(len(shades), 2)
        self.assertEqual(shades[0]["undertone"], "quente")
        self.assertEqual(shades[1]["shade_hex"], "#a87565")
        self.assertFalse(shades[1]["available"])

    def test_extracts_html_entity_encoded_variant_hex(self):
        document = (
            "<selector :variants='[{&quot;value&quot;:&quot;C4 Cool&quot;,"
            "&quot;hex&quot;:&quot;#B98268&quot;}]'></selector>"
        )
        shades = _discovered_shades(document)
        self.assertEqual(shades[0]["shade_hex"], "#b98268")
        self.assertEqual(shades[0]["undertone"], "frio")

    def test_catalog_has_broad_coverage(self):
        sources = json.loads((PROJECT_ROOT / "dev" / "product_sources.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(sources), 15)
        self.assertGreaterEqual(len({source["source"] for source in sources}), 10)

    def test_estimates_uniform_official_swatch(self):
        image = Image.new("RGB", (80, 80), "white")
        for x in range(10, 70):
            for y in range(10, 70):
                image.putpixel((x, y), (201, 145, 105))
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        shade_hex, confidence = _estimate_swatch_hex(payload.getvalue())
        self.assertEqual(shade_hex, "#c99169")
        self.assertGreaterEqual(confidence, 0.75)

    def test_collect_falls_back_to_authorized_retailer(self):
        source = {"url": "https://brand.test/product", "fallback_urls": ["https://retailer.test/product"]}
        with patch("product_sync._collect_one", side_effect=[ValueError("403"), {"data_source": "retailer.test"}]) as mocked:
            result = collect(source)
        self.assertEqual(result["data_source"], "retailer.test")
        self.assertEqual(mocked.call_count, 2)

    def test_mufe_screenshot_catalog_tracks_shade_availability(self):
        sources = json.loads((PROJECT_ROOT / "dev" / "product_sources.json").read_text(encoding="utf-8"))
        source = next(item for item in sources if item.get("external_id") == "I000075100")
        self.assertEqual(len(source["shades"]), 37)
        self.assertEqual(sum(bool(shade.get("available")) for shade in source["shades"]), 7)
        self.assertEqual(source["price"], 49)
        with patch("product_sync._collect_one", side_effect=ValueError("403")):
            record = collect(source)
        shade = next(item for item in record["shades"] if item["shade_code"] == "4R76")
        self.assertEqual(
            shade["shade_url"],
            "https://www.makeupforever.com/us/en/face/foundation/hd-skin-I000075176.html",
        )

    def test_huda_shopify_variant_links_select_the_requested_shade(self):
        sources = json.loads((PROJECT_ROOT / "dev" / "product_sources.json").read_text(encoding="utf-8"))
        source = next(item for item in sources if item.get("external_id") == "P512640")
        with patch("product_sync._collect_one", side_effect=ValueError("403")):
            source = {**source, "allow_static_fallback": True}
            record = collect(source)
        self.assertEqual(
            record["shades"][0]["shade_url"],
            "https://hudabeauty.com/en-us/products/easy-blur-natural-airbrush-foundation-with-niacinamide-hb01166m?variant=50573349552406",
        )
        self.assertEqual(
            record["shades"][-1]["shade_url"],
            "https://hudabeauty.com/en-us/products/easy-blur-natural-airbrush-foundation-with-niacinamide-hb01166m?variant=50573350764822",
        )

    def test_static_screenshot_fallback_survives_blocked_page(self):
        source = {
            "url": "https://brand.test/product", "external_id": "sku", "brand": "Brand",
            "name": "Product", "price": 49, "currency": "USD", "allow_static_fallback": True,
            "shades": [{"name": "N1", "shade_hex": "#ccbbaa", "available": False}],
        }
        with patch("product_sync._collect_one", side_effect=ValueError("403")):
            result = collect(source)
        self.assertEqual(result["price"], 49)
        self.assertFalse(result["shades"][0]["available"])
        self.assertEqual(result["shades"][0]["shade_hex_source"], "curated_screenshot_estimate")

    def test_hydra_glow_sizes_have_separate_prices_and_inventory(self):
        sources = json.loads((PROJECT_ROOT / "dev" / "product_sources.json").read_text(encoding="utf-8"))
        standard = next(item for item in sources if item.get("external_id") == "P510064")
        mini = next(item for item in sources if item.get("external_id") == "P510064-MINI")
        self.assertEqual((standard["price"], standard["currency"], len(standard["shades"])), (64.5, "CAD", 32))
        self.assertEqual((mini["price"], mini["currency"], len(mini["shades"])), (31.5, "CAD", 6))
        self.assertFalse(next(shade for shade in standard["shades"] if shade["shade_code"] == "2Y32")["available"])
        self.assertFalse(next(shade for shade in mini["shades"] if shade["shade_code"] == "1N00")["available"])
        with patch("product_sync._collect_one", side_effect=ValueError("403")):
            standard_record = collect(standard)
            mini_record = collect(mini)
        self.assertEqual(
            next(shade for shade in standard_record["shades"] if shade["shade_code"] == "4R76")["shade_url"],
            "https://www.makeupforever.com/us/en/face/foundation/hd-skin-hydra-glow-I000062476.html",
        )
        self.assertEqual(
            next(shade for shade in mini_record["shades"] if shade["shade_code"] == "4N62")["shade_url"],
            "https://www.makeupforever.com/us/en/face/foundation/hd-skin-hydra-glow---mini-I000061862.html",
        )

    def test_matte_velvet_screenshot_catalog(self):
        sources = json.loads((PROJECT_ROOT / "dev" / "product_sources.json").read_text(encoding="utf-8"))
        powder = next(item for item in sources if item.get("external_id") == "P504432")
        self.assertEqual((powder["price"], powder["currency"]), (60, "CAD"))
        self.assertEqual(len(powder["shades"]), 22)
        self.assertTrue(all(shade["available"] for shade in powder["shades"]))
        with patch("product_sync._collect_one", side_effect=ValueError("403")):
            record = collect(powder)
        self.assertEqual(
            next(shade for shade in record["shades"] if shade["shade_code"] == "4N74")["shade_url"],
            "https://www.makeupforever.com/us/en/face/foundation/hd-skin-matte-velvet-I000064474.html",
        )


if __name__ == "__main__":
    unittest.main()
