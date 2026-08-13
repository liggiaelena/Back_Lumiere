import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dev"))

from app.gpt_recommendations import RecommendationUnavailableError
from app.recommendation_service import AllRecommendationStrategiesFailed, recommend_with_fallback


ARGS = {
    "skin_hex": "#C68B6E",
    "fitzpatrick": 4,
    "undertone": "neutral",
}


class RecommendationStrategyTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.recommendation_service.recommend_products", new_callable=AsyncMock)
    async def test_plan_one_is_preferred(self, web_search):
        web_search.return_value = {
            "shades": [{"brand": "Live"}], "reliable": True,
            "catalog_source": "openai_web_search", "catalog_shades_considered": None,
            "search_summary": "live", "model": "test-model",
        }
        result = await recommend_with_fallback(**ARGS)
        self.assertEqual(result["strategy"], "plan_1_openai_web_search")
        self.assertFalse(result["fallback_used"])

    @patch("app.recommendation_service.get_catalog_recommendations")
    @patch("app.recommendation_service.recommend_products", new_callable=AsyncMock)
    async def test_plan_two_runs_after_plan_one_failure(self, web_search, catalog):
        web_search.side_effect = RecommendationUnavailableError("web unavailable")
        catalog.return_value = {
            "shades": [{"brand": "Stored"}], "reliable": True,
            "catalog_source": "neon_postgres", "catalog_shades_considered": 20,
        }
        result = await recommend_with_fallback(**ARGS)
        self.assertEqual(result["strategy"], "plan_2_neon_catalog")
        self.assertTrue(result["fallback_used"])
        self.assertIn("web unavailable", result["primary_error"])

    @patch("app.recommendation_service.get_catalog_recommendations")
    @patch("app.recommendation_service.recommend_products", new_callable=AsyncMock)
    async def test_force_fallback_skips_plan_one(self, web_search, catalog):
        catalog.return_value = {
            "shades": [{"brand": "Stored"}], "reliable": True,
            "catalog_source": "neon_postgres", "catalog_shades_considered": 20,
        }
        result = await recommend_with_fallback(**ARGS, force_fallback=True)
        web_search.assert_not_awaited()
        self.assertEqual(result["strategy"], "plan_2_neon_catalog")
        self.assertTrue(result["fallback_used"])

    @patch("app.recommendation_service.get_catalog_recommendations")
    @patch("app.recommendation_service.recommend_products", new_callable=AsyncMock)
    async def test_both_fail_without_inventing_results(self, web_search, catalog):
        web_search.side_effect = RecommendationUnavailableError("web unavailable")
        catalog.side_effect = RuntimeError("database unavailable")
        with self.assertRaises(AllRecommendationStrategiesFailed):
            await recommend_with_fallback(**ARGS)


if __name__ == "__main__":
    unittest.main()
