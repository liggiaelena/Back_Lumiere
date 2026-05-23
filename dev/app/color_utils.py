def hex_to_rgb(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def color_delta(hex1: str, hex2: str) -> float:
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    return ((0.299*(r1-r2))**2 + (0.587*(g1-g2))**2 + (0.114*(b1-b2))**2)**0.5

from app.recommendations import get_recommendations

def _average_hex(hex_list: list) -> str:
    rgbs = [hex_to_rgb(h) for h in hex_list if h and len(h) == 7]
    if not rgbs:
        return "#c68b6e"
    r = int(sum(c[0] for c in rgbs) / len(rgbs))
    g = int(sum(c[1] for c in rgbs) / len(rgbs))
    b = int(sum(c[2] for c in rgbs) / len(rgbs))
    return f"#{r:02x}{g:02x}{b:02x}"

# Canonical mid-point hex for each Fitzpatrick type (well-established references)
_FITZPATRICK_HEX = {
    1: "#f6ede4",
    2: "#f3d7c0",
    3: "#dba98a",
    4: "#b07d5b",
    5: "#7d4e2d",
    6: "#3e1f0e",
}

def hex_to_fitzpatrick(hex_color: str) -> int:
    """Map a hex color to the closest Fitzpatrick type by weighted color distance."""
    return min(
        _FITZPATRICK_HEX,
        key=lambda fitz: color_delta(hex_color, _FITZPATRICK_HEX[fitz])
    )

def build_final_report(region_results: dict, skin_tone: dict | None = None) -> dict:
    regioes = region_results

    tons_fitz = [r["tom_fitzpatrick"] for r in regioes.values()]
    claude_fitzpatrick = max(set(tons_fitz), key=tons_fitz.count)
    tom_geral_hex = _average_hex([r.get("tom_hex", "") for r in regioes.values()])

    subtoms = [r["subtom"] for r in regioes.values()]
    subtom_geral = max(set(subtoms), key=subtoms.count)

    # Use BiSeNet's precise median hex to determine Fitzpatrick when available;
    # fall back to Claude's majority vote.
    bisenet_hex = skin_tone.get("median_hex") if skin_tone else None
    if bisenet_hex:
        tom_geral = hex_to_fitzpatrick(bisenet_hex)
        fitzpatrick_source = "bisenet"
    else:
        tom_geral = claude_fitzpatrick
        fitzpatrick_source = "claude"

    pares = [
        ("testa",  "bochecha_e"),
        ("testa",  "bochecha_d"),
        ("nariz",  "bochecha_e"),
        ("queixo", "testa"),
    ]

    comparacoes = {}
    for r1, r2 in pares:
        if r1 in regioes and r2 in regioes:
            delta = color_delta(regioes[r1]["tom_hex"], regioes[r2]["tom_hex"])
            comparacoes[f"{r1}_vs_{r2}"] = {
                "delta": round(delta, 2),
                "nivel": "alto" if delta > 20 else "moderado" if delta > 10 else "baixo"
            }

    todas_imperf = []
    for region, data in regioes.items():
        for imp in data.get("imperfeicoes", []):
            todas_imperf.append({**imp, "regiao": region})

    recommendations = get_recommendations(tom_geral, subtom_geral, skin_hex=bisenet_hex)

    return {
        "tom_geral_fitzpatrick":  tom_geral,
        "tom_geral_hex":          bisenet_hex or tom_geral_hex,
        "fitzpatrick_source":     fitzpatrick_source,
        "subtom_predominante":    subtom_geral,
        "regioes":                regioes,
        "comparacao_tons":        comparacoes,
        "imperfeicoes":           todas_imperf,
        "recommendations":        recommendations,
        "skin_tone":              skin_tone,
    }
