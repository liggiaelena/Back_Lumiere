"""Opt-in live latency benchmark for OpenAI product recommendations.

Run explicitly because it performs two billable web-search requests:

    RUN_OPENAI_LATENCY_TESTS=1 .venv/bin/python -m unittest \
        tests.test_recommendation_latency -v
"""

import os
import sys
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dev"))

from app.config import settings
from app.gpt_recommendations import recommend_products


RUN_LIVE = os.getenv("RUN_OPENAI_LATENCY_TESTS") == "1"


def _assert_quality(test_case: unittest.TestCase, result: dict) -> None:
    """Check the fields the UI and recommendation safety filters depend on."""
    shades = result["shades"]
    test_case.assertGreaterEqual(len(shades), 3)
    test_case.assertLessEqual(len(shades), 8)
    for product in shades:
        for field in (
            "brand", "product_name", "shade_name", "shade_code", "shade_hex",
            "undertone", "product_url", "recommendation_reason", "source_urls",
        ):
            test_case.assertTrue(product[field], f"missing {field}")
        test_case.assertTrue(product["product_url"].startswith(("http://", "https://")))
        test_case.assertTrue(all(url.startswith(("http://", "https://")) for url in product["source_urls"]))


@unittest.skipUnless(RUN_LIVE, "set RUN_OPENAI_LATENCY_TESTS=1 to run billable live benchmark")
class RecommendationLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_baseline_vs_latency_optimized(self):
        self.assertTrue(settings.openai_api_key, "OPENAI_API_KEY is required")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        common = {
            "skin_hex": "#B38976",
            "fitzpatrick": 3,
            "undertone": "warm (golden, peachy, yellowish)",
            "condition_map": {},
            "excluded_allergens": [],
            "lang": "en",
            "client": client,
        }

        measurements = {}
        results = {}
        for name, optimized in (("baseline", False), ("optimized", True)):
            started = time.perf_counter()
            results[name] = await recommend_products(
                **common,
                latency_optimized=optimized,
            )
            measurements[name] = time.perf_counter() - started
            _assert_quality(self, results[name])
            print(
                f"\n{name}: {measurements[name]:.2f}s, "
                f"products={len(results[name]['shades'])}"
            )

        improvement = measurements["baseline"] - measurements["optimized"]
        percent = improvement / measurements["baseline"] * 100
        print(
            f"\nlatency improvement: {improvement:.2f}s ({percent:.1f}%)\n"
            f"baseline={measurements['baseline']:.2f}s "
            f"optimized={measurements['optimized']:.2f}s"
        )


if __name__ == "__main__":
    unittest.main()
