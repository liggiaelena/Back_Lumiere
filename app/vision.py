import anthropic, json, asyncio
from app.config import settings

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

PROMPT = """You are a skin analysis expert. Analyze this image of an isolated facial region and respond ONLY with valid JSON, no additional text, no markdown, no explanations.

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
  "notas": "observation in English, maximum 1 sentence"
}

Fitzpatrick scale: 1=very light, 2=light, 3=medium light, 4=medium, 5=dark, 6=very dark.
If no visible imperfections, return imperfeicoes as empty list.
Analyze only what is visible — do not invent data."""

async def analyze_region(region_name: str, b64_crop: str) -> dict:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _client.messages.create(
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
                            "data": b64_crop
                        }
                    },
                    {"type": "text", "text": PROMPT}
                ]
            }]
        )
    )
    raw = result.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "tom_hex": "#c68b6e",
            "tom_fitzpatrick": 3,
            "subtom": "neutro",
            "oleosidade": "normal",
            "imperfeicoes": [],
            "uniformidade": 5,
            "notas": "Analysis unavailable for this region.",
        }
