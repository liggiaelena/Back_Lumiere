import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dev"))

from app.gpt_recommendations import RecommendationUnavailableError, recommend_products


def _product(**overrides):
    value = {
        "brand": "Example Beauty", "product_name": "True Match Foundation",
        "shade_name": "Neutral Beige", "shade_code": "N30",
        "shade_hex": "#C58F72", "undertone": "neutral", "price": 49.0,
        "currency": "USD", "ingredients": ["water", "glycerin"],
        "allergens": [],
        "product_url": "https://example.com/products/foundation?shade=N30",
        "image_url": None, "data_source": "Example Beauty",
        "recommendation_reason": "Close shade and undertone match.",
        "source_urls": ["https://example.com/products/foundation?shade=N30"],
    }
    value.update(overrides)
    return value


class _Responses:
    def __init__(self, payload):
        self.payload, self.kwargs = payload, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return type("Response", (), {"output_text": json.dumps(self.payload)})()


class _Client:
    def __init__(self, payload):
        self.responses = _Responses(payload)


class GptRecommendationsTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_web_search_and_returns_frontend_contract(self):
        client = _Client({"recommendations": [_product()], "search_summary": "Verified live."})
        result = await recommend_products(
            skin_hex="#C68B6E", fitzpatrick=4, undertone="neutral", client=client
        )
        self.assertEqual(client.responses.kwargs["tools"], [{"type": "web_search"}])
        self.assertEqual(result["catalog_source"], "openai_web_search")
        self.assertEqual(result["shades"][0]["where_to_buy"], _product()["product_url"])

    async def test_excluded_ingredient_is_removed_fail_closed(self):
        client = _Client({"recommendations": [_product(ingredients=["water", "fragrance"])], "search_summary": "Checked."})
        with self.assertRaises(RecommendationUnavailableError):
            await recommend_products(
                skin_hex="#C68B6E", fitzpatrick=4, undertone="neutral",
                excluded_allergens=["fragrance"], client=client,
            )

    async def test_unverifiable_product_url_is_rejected(self):
        client = _Client({"recommendations": [_product(product_url="not-a-url")], "search_summary": "No link."})
        with self.assertRaises(RecommendationUnavailableError):
            await recommend_products(
                skin_hex="#C68B6E", fitzpatrick=4, undertone="neutral", client=client
            )


if __name__ == "__main__":
    unittest.main()
