"""Web-grounded cosmetic recommendations using the OpenAI Responses API."""

import json
import logging
from typing import Any
from urllib.parse import urlparse

from app.config import settings

logger = logging.getLogger(__name__)


DEFAULT_ALLERGEN_OPTIONS = [
    "almond", "benzyl alcohol", "benzyl benzoate", "benzyl salicylate",
    "cinnamal", "citral", "coumarin", "eugenol", "fragrance", "geraniol",
    "isoeugenol", "lanolin", "latex", "limonene", "linalool",
    "methylisothiazolinone", "oakmoss", "phenoxyethanol", "propylene glycol",
    "soy", "wheat",
]


class RecommendationUnavailableError(RuntimeError):
    """Raised when web recommendations cannot be produced safely."""


_PRODUCT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "brand": {"type": "string"},
        "product_name": {"type": "string"},
        "shade_name": {"type": "string"},
        "shade_code": {"type": "string"},
        "shade_hex": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
        "undertone": {"type": "string"},
        "price": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "ingredients": {"type": "array", "items": {"type": "string"}},
        "allergens": {"type": "array", "items": {"type": "string"}},
        "product_url": {"type": "string"},
        "image_url": {"type": ["string", "null"]},
        "data_source": {"type": "string"},
        "recommendation_reason": {"type": "string"},
        "source_urls": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "brand", "product_name", "shade_name", "shade_code", "shade_hex",
        "undertone", "price", "currency", "ingredients", "allergens",
        "product_url", "image_url", "data_source", "recommendation_reason",
        "source_urls",
    ],
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recommendations": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": _PRODUCT_SCHEMA,
        },
        "search_summary": {"type": "string"},
    },
    "required": ["recommendations", "search_summary"],
}


def _is_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalise_product(product: dict, excluded: set[str]) -> dict | None:
    product_url = product.get("product_url")
    sources = [url for url in product.get("source_urls", []) if _is_http_url(url)]
    if not _is_http_url(product_url) or not sources:
        return None

    allergens = [str(value).strip().casefold() for value in product.get("allergens", []) if str(value).strip()]
    ingredients = [str(value).strip() for value in product.get("ingredients", []) if str(value).strip()]
    # When exclusions are active, missing ingredient evidence is unsafe.
    searchable_ingredients = " | ".join(value.casefold() for value in ingredients)
    ingredient_matches = {value for value in excluded if value in searchable_ingredients}
    if excluded and (not ingredients or excluded.intersection(allergens) or ingredient_matches):
        return None

    price = product.get("price")
    currency = product.get("currency")
    price_range = None
    if isinstance(price, (int, float)):
        price_range = f"{currency or ''} {price:.2f}".strip()

    return {
        **product,
        "allergens": allergens,
        "ingredients": ingredients,
        "price_range": price_range,
        "where_to_buy": product_url,
        "shade_hex_source": "gpt_web_search",
        "shade_hex_confidence": None,
        "source_urls": sources,
    }


async def recommend_products(
    *,
    skin_hex: str,
    fitzpatrick: int,
    undertone: str,
    condition_map: dict | None = None,
    excluded_allergens: list[str] | None = None,
    lang: str = "en",
    client=None,
) -> dict:
    """Search the live web and return structured, source-backed products."""
    if client is None:
        if not settings.openai_api_key:
            raise RecommendationUnavailableError("OPENAI_API_KEY is not configured")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RecommendationUnavailableError("The openai package is not installed") from exc
        client = AsyncOpenAI(api_key=settings.openai_api_key)

    exclusions = sorted({value.strip().casefold() for value in (excluded_allergens or []) if value.strip()})
    conditions = [
        name for name, details in (condition_map or {}).items()
        if details is True or (isinstance(details, dict) and details.get("detected") is True)
    ]
    prompt = f"""
Find current foundation or complexion products on reputable brand and major
cosmetics retailer websites for this measured skin profile:
- skin HEX: {skin_hex}
- Fitzpatrick type: {fitzpatrick}
- undertone: {undertone}
- detected visible conditions: {conditions or ['none']}
- excluded allergens/ingredients: {exclusions or ['none']}
- response language: {lang}

Use web search. Recommend 3-8 currently purchasable products with an exact shade
name/code suitable for the measured color. Prefer a URL that opens the exact
shade variant; otherwise use the product page and say so in the reason. Verify
current price, currency, availability, shade, ingredients/allergens, and URL
from search sources. Never invent a product, shade, price, ingredient list, or
URL. If allergen exclusions are present, omit products without enough ingredient
evidence and omit every product containing an excluded item. `source_urls` must
contain the pages used to verify each recommendation. This is cosmetic guidance,
not medical advice.
""".strip()

    try:
        response = await client.responses.create(
            model=settings.openai_recommendation_model,
            tools=[{"type": "web_search"}],
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "cosmetic_recommendations",
                    "strict": True,
                    "schema": _RESPONSE_SCHEMA,
                }
            },
        )
        payload = json.loads(response.output_text)
    except Exception as exc:
        logger.exception("OpenAI web recommendation request failed")
        raise RecommendationUnavailableError("OpenAI web recommendation request failed") from exc

    excluded_set = set(exclusions)
    recommendations = [
        normalised
        for product in payload.get("recommendations", [])
        if (normalised := _normalise_product(product, excluded_set)) is not None
    ]
    if not recommendations:
        raise RecommendationUnavailableError("Web search returned no verifiable products")

    return {
        "shades": recommendations,
        "reliable": True,
        "catalog_source": "openai_web_search",
        "catalog_shades_considered": None,
        "search_summary": payload.get("search_summary", ""),
        "model": settings.openai_recommendation_model,
    }
