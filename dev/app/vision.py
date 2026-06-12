import json
import asyncio
from typing import Optional

from app.config import settings


# Lazy client holders
_anthropic_client: Optional[object] = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic

            _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        except Exception:
            _anthropic_client = None
    return _anthropic_client

PROMPT = """You are an image analysis assistant for a non-clinical academic prototype. Analyze this image of an isolated facial region and respond ONLY with valid JSON, no additional text, no markdown, no explanations.

Required structure:
{
  "tom_hex": "#RRGGBB",
  "tom_fitzpatrick": number from 1 to 6,
  "subtom": "quente" | "frio" | "neutro",
  "oleosidade": "seco" | "normal" | "misto" | "oleoso",
  "imperfeicoes": [
    {"tipo": "acne"|"mancha"|"poro"|"linha"|"vermelhidao"|"outro", "intensidade": "leve"|"moderado"|"intenso"}
  ],
  "uniformidade": number from 0 to 10,
  "condition_map": {
    "melanoma_suspected": {
      "confidence": number from 0.0 to 1.0,
      "reason": "short non-diagnostic reason"
    }
  },
  "notas": "observation in English, maximum 1 sentence"
}

Fitzpatrick scale: 1=very light, 2=light, 3=medium light, 4=medium, 5=dark, 6=very dark.
If no visible imperfections, return imperfeicoes as empty list.

Do not diagnose medical conditions. This is not a clinical diagnosis system.
Only set melanoma_suspected confidence above 0.85 if there is a clearly visible suspicious lesion pattern that should be escalated for medical review.
Otherwise keep melanoma_suspected confidence low.
Analyze only what is visible — do not invent data."""

async def analyze_region(region_name: str, b64_crop: str) -> dict:
    loop = asyncio.get_event_loop()
    provider = settings.provider

    if provider == "anthropic":
        client = _get_anthropic_client()
        if client is None:
            # Anthropic SDK not available or key missing
            return _fallback_response()

        result = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-opus-4-5",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_crop,
                            }
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }],
            ),
        )
        raw = result.content[0].text.strip()
    elif provider == "gemini":
        # Gemini integration not implemented yet; return fallback.
        # Option: implement Google Generative API calls here if desired.
        return _fallback_response()
    else:
        return _fallback_response()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_response()


def _fallback_response() -> dict:
    return {
        "tom_hex": "#c68b6e",
        "tom_fitzpatrick": 3,
        "subtom": "neutro",
        "oleosidade": "normal",
        "imperfeicoes": [],
        "uniformidade": 5,
        "condition_map": {
            "melanoma_suspected": {
                "confidence": 0.0,
                "reason": "Analysis unavailable for this region.",
            }
        },
        "notas": "Analysis unavailable for this region.",
    }
