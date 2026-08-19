import json
import logging
import asyncio
from typing import Optional, Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)
_region_semaphore = None


def _get_region_semaphore():
    global _region_semaphore
    if _region_semaphore is None:
        _region_semaphore = asyncio.Semaphore(settings.openai_region_concurrency)
    return _region_semaphore

_VALID_SUBTOM = {"quente", "frio", "neutro"}
_VALID_OLEOSIDADE = {"seco", "normal", "misto", "oleoso"}


def _validate_region_response(data: Any) -> bool:
    """Check that OpenAI's JSON actually has the fields/types the prompt
    asked for, instead of accepting any parseable JSON at face value."""
    if not isinstance(data, dict):
        return False

    if not isinstance(data.get("tom_hex"), str) or not data["tom_hex"].startswith("#"):
        return False
    if not isinstance(data.get("tom_fitzpatrick"), (int, float)) or not (1 <= data["tom_fitzpatrick"] <= 6):
        return False
    if data.get("subtom") not in _VALID_SUBTOM:
        return False
    if data.get("oleosidade") not in _VALID_OLEOSIDADE:
        return False
    if not isinstance(data.get("imperfeicoes"), list):
        return False
    if not isinstance(data.get("uniformidade"), (int, float)) or not (0 <= data["uniformidade"] <= 10):
        return False
    if not isinstance(data.get("condition_map"), dict):
        return False
    if not isinstance(data.get("notas"), dict):
        return False

    return True


# Lazy OpenAI client holder
_openai_client: Optional[object] = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=settings.openai_api_key)
        except Exception as exc:
            logger.error("OpenAI client creation failed: %s", exc)
            _openai_client = None
    return _openai_client


REGION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tom_hex": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
        "tom_fitzpatrick": {"type": "number", "minimum": 1, "maximum": 6},
        "subtom": {"type": "string", "enum": ["quente", "frio", "neutro"]},
        "oleosidade": {"type": "string", "enum": ["seco", "normal", "misto", "oleoso"]},
        "imperfeicoes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tipo": {"type": "string", "enum": ["acne", "mancha", "poro", "linha", "vermelhidao", "outro"]},
                    "intensidade": {"type": "string", "enum": ["leve", "moderado", "intenso"]},
                },
                "required": ["tipo", "intensidade"],
            },
        },
        "uniformidade": {"type": "number", "minimum": 0, "maximum": 10},
        "condition_map": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "melanoma_suspected": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["confidence", "reason"],
                }
            },
            "required": ["melanoma_suspected"],
        },
        "notas": {
            "type": "object",
            "additionalProperties": False,
            "properties": {code: {"type": "string"} for code in ("en", "tw", "zh", "pt", "fr", "tr")},
            "required": ["en", "tw", "zh", "pt", "fr", "tr"],
        },
    },
    "required": ["tom_hex", "tom_fitzpatrick", "subtom", "oleosidade", "imperfeicoes", "uniformidade", "condition_map", "notas"],
}

BASE_PROMPT = """You are an image analysis assistant for a non-clinical academic prototype. Analyze this image of an isolated facial region and respond ONLY with valid JSON, no additional text, no markdown, no explanations.

Region name:
{region_name}

SegFormer condition context:
{condition_context}

How to use SegFormer context:
- Use the SegFormer condition context as supporting evidence only.
- You must still inspect the visible facial region image.
- Do not blindly copy the SegFormer result if this crop does not visually support it.
- Do not diagnose medical conditions. This is not a clinical diagnosis system.
- If SegFormer says a condition may exist outside this crop, do not claim it is visible in this crop unless you can see supporting evidence.

Required structure:
{{
  "tom_hex": "#RRGGBB",
  "tom_fitzpatrick": number from 1 to 6,
  "subtom": "quente" | "frio" | "neutro",
  "oleosidade": "seco" | "normal" | "misto" | "oleoso",
  "imperfeicoes": [
    {{"tipo": "acne"|"mancha"|"poro"|"linha"|"vermelhidao"|"outro", "intensidade": "leve"|"moderado"|"intenso"}}
  ],
  "uniformidade": number from 0 to 10,
  "condition_map": {{
    "melanoma_suspected": {{
      "confidence": number from 0.0 to 1.0,
      "reason": "short non-diagnostic reason"
    }}
  }},
  "notas": {{
    "en": "observation in English, maximum 1 sentence",
    "tw": "observation in Traditional Chinese, maximum 1 sentence",
    "zh": "observation in Simplified Chinese, maximum 1 sentence",
    "pt": "observation in Portuguese, maximum 1 sentence",
    "fr": "observation in French, maximum 1 sentence",
    "tr": "observation in Turkish, maximum 1 sentence"
  }}
}}

Fitzpatrick scale: 1=very light, 2=light, 3=medium light, 4=medium, 5=dark, 6=very dark.
If no visible imperfections, return imperfeicoes as empty list.

Only set melanoma_suspected confidence above 0.85 if there is a clearly visible suspicious lesion pattern that should be escalated for medical review.
Otherwise keep melanoma_suspected confidence low.
Analyze only what is visible — do not invent data."""


LANGUAGE_MAP = {
    "en": "English",
    "tw": "Traditional Chinese",
    "zh": "Simplified Chinese",
    "pt": "Portuguese",
    "fr": "French",
    "tr": "Turkish",
}


def _build_prompt(
    region_name: str,
    condition_map: Optional[Dict[str, Any]] = None,
    lang: str = "en",
) -> str:
    """
    Build the region-analysis prompt with SegFormer condition_map as context.

    condition_map is generated by app.segmentation.get_condition_outputs().
    Example:
        {
            "vitiligo": {"detected": False, "area_percent": 0, "zones": []},
            "melasma": {"detected": True, "area_percent": 3.2, "zones": ["left_cheek"]},
            "wine_stain": {"detected": False, "area_percent": 0, "zones": []}
        }
    """

    if condition_map:
        condition_context = json.dumps(
            condition_map,
            ensure_ascii=False,
            indent=2,
        )
    else:
        condition_context = json.dumps(
            {
                "vitiligo": {
                    "detected": False,
                    "area_percent": 0,
                    "zones": [],
                },
                "melasma": {
                    "detected": False,
                    "area_percent": 0,
                    "zones": [],
                },
                "wine_stain": {
                    "detected": False,
                    "area_percent": 0,
                    "zones": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    return BASE_PROMPT.format(
        region_name=region_name,
        condition_context=condition_context,
    )


async def analyze_region(
    region_name: str,
    b64_crop: str,
    condition_map: Optional[Dict[str, Any]] = None,
    lang: str = "en",
) -> dict:
    """
    Analyze one facial region crop.

    condition_map is optional for backward compatibility.
    pipeline.py will pass the SegFormer output here as model context.
    """

    loop = asyncio.get_event_loop()
    provider = settings.provider

    prompt = _build_prompt(
        region_name=region_name,
        condition_map=condition_map,
        lang=lang,
    )

    if provider == "openai":
        client = _get_openai_client()

        if client is None:
            logger.error("OpenAI client is None, returning fallback response.")
            return fallback_response()

        try:
            async with _get_region_semaphore():
                result = await loop.run_in_executor(
                    None,
                    lambda: client.responses.create(
                    model=settings.openai_region_model,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_crop}",
                                "detail": "high",
                            },
                        ],
                    }],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "region_analysis",
                            "strict": True,
                            "schema": REGION_RESPONSE_SCHEMA,
                        }
                    },
                    ),
                )
        except Exception as exc:
            logger.error("OpenAI region analysis failed, using fallback response: %s", exc)
            return fallback_response()

        raw = result.output_text.strip()

    else:
        return fallback_response()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("OpenAI response for region '%s' was not valid JSON: %r", region_name, raw[:200])
        return fallback_response()

    if not _validate_region_response(parsed):
        logger.warning("OpenAI response for region '%s' failed schema validation: %r", region_name, parsed)
        return fallback_response()

    return parsed


def fallback_response() -> dict:
    """Placeholder used when no real analysis could be obtained. Callers
    MUST check `analysis_unavailable` before treating these values as real
    measurements — they are not derived from the image at all."""
    return {
        "tom_hex": "#c68b6e",
        "tom_fitzpatrick": 3,
        "subtom": "neutro",
        "oleosidade": "normal",
        "imperfeicoes": [],
        "uniformidade": 5,
        "analysis_unavailable": True,
        "condition_map": {
            "melanoma_suspected": {
                "confidence": 0.0,
                "reason": "Analysis unavailable for this region.",
            }
        },
        "notas": {
            "en": "Analysis unavailable for this region.",
            "tw": "該區域暫無分析數據。",
            "zh": "该区域暂无分析数据。",
            "pt": "Análise indisponível para esta região.",
            "fr": "Analyse non disponible pour cette région.",
            "tr": "Bu bölge için analiz mevcut değil."
        },
    }
