import anthropic, json, asyncio
from app.config import settings

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

PROMPT = """Você é um especialista em análise de pele. Analise esta imagem de uma região facial isolada e responda APENAS com JSON válido, sem texto adicional, sem markdown, sem explicações.

Estrutura obrigatória:
{
  "tom_hex": "#RRGGBB",
  "tom_fitzpatrick": número de 1 a 6,
  "subtom": "quente" | "frio" | "neutro",
  "oleosidade": "seco" | "normal" | "misto" | "oleoso",
  "imperfeicoes": [
    {"tipo": "acne"|"mancha"|"poro"|"linha"|"vermelhidao"|"outro", "intensidade": "leve"|"moderado"|"intenso"}
  ],
  "uniformidade": número de 0 a 10,
  "notas": "observação em português, máximo 1 frase"
}

Escala Fitzpatrick: 1=muito claro, 2=claro, 3=médio claro, 4=médio, 5=escuro, 6=muito escuro.
Se não houver imperfeições visíveis, retorne imperfeicoes como lista vazia.
Analise apenas o que é visível — não invente dados."""

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
    return json.loads(raw)
