"""Two-stage recommendation strategy: live web first, Neon catalogue second."""

import asyncio
import logging

from app.fallback_catalog.recommendations import get_recommendations as get_catalog_recommendations
from app.gpt_recommendations import RecommendationUnavailableError, recommend_products

logger = logging.getLogger(__name__)


class AllRecommendationStrategiesFailed(RuntimeError):
    """Raised when neither the primary nor fallback strategy can return products."""


async def recommend_with_fallback(
    *,
    skin_hex: str,
    fitzpatrick: int,
    undertone: str,
    condition_map: dict | None = None,
    excluded_allergens: list[str] | None = None,
    lang: str = "en",
) -> dict:
    try:
        result = await recommend_products(
            skin_hex=skin_hex,
            fitzpatrick=fitzpatrick,
            undertone=undertone,
            condition_map=condition_map,
            excluded_allergens=excluded_allergens,
            lang=lang,
        )
        return {
            **result,
            "strategy": "plan_1_openai_web_search",
            "fallback_used": False,
            "primary_error": None,
        }
    except RecommendationUnavailableError as primary_exc:
        logger.warning("Plan 1 failed; trying Neon catalogue fallback: %s", primary_exc)
        primary_error = str(primary_exc)

    try:
        result = await asyncio.to_thread(
            get_catalog_recommendations,
            fitzpatrick,
            undertone,
            skin_hex,
            3,
            condition_map,
            excluded_allergens,
        )
        if not result.get("shades"):
            raise RuntimeError("Neon catalogue returned no matching products")
        return {
            **result,
            "search_summary": "Live web search failed; results came from the stored Neon catalogue.",
            "model": None,
            "strategy": "plan_2_neon_catalog",
            "fallback_used": True,
            "primary_error": primary_error,
        }
    except Exception as fallback_exc:
        logger.exception("Plan 2 Neon catalogue fallback failed")
        raise AllRecommendationStrategiesFailed(
            f"Plan 1 failed: {primary_error}; Plan 2 failed: {fallback_exc}"
        ) from fallback_exc
