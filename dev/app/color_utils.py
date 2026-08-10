import math


def hex_to_rgb(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def color_delta(hex1: str, hex2: str) -> float:
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    return ((0.299*(r1-r2))**2 + (0.587*(g1-g2))**2 + (0.114*(b1-b2))**2)**0.5

from app.medical_alert import apply_medical_triage, build_condition_map_from_regions

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
    """Map a skin-tone hex color to an approximate Fitzpatrick type.

    Skin tone should track perceived lightness more than the nearest RGB swatch.
    ITA-style Lab thresholds are less sensitive to redness from conditions such
    as port-wine stains and to small platform color-management differences.
    """
    ita = _hex_to_ita(hex_color)

    if ita is not None:
        if ita > 55:
            return 1
        if ita > 41:
            return 2
        if ita > 28:
            return 3
        if ita > 10:
            return 4
        if ita > -30:
            return 5
        return 6

    return min(
        _FITZPATRICK_HEX,
        key=lambda fitz: color_delta(hex_color, _FITZPATRICK_HEX[fitz])
    )


def _hex_to_ita(hex_color: str) -> float | None:
    try:
        r, g, b = hex_to_rgb(hex_color)
    except (TypeError, ValueError):
        return None

    l_star, _, b_star = _rgb_to_lab(r, g, b)

    if b_star == 0:
        return 90.0 if l_star >= 50 else -90.0

    return math.degrees(math.atan((l_star - 50.0) / b_star))


def _rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    r_lin = _srgb_to_linear(r / 255.0)
    g_lin = _srgb_to_linear(g / 255.0)
    b_lin = _srgb_to_linear(b / 255.0)

    x = (0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin) / 0.95047
    y = (0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin) / 1.00000
    z = (0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin) / 1.08883

    fx = _lab_f(x)
    fy = _lab_f(y)
    fz = _lab_f(z)

    return (
        116.0 * fy - 16.0,
        500.0 * (fx - fy),
        200.0 * (fy - fz),
    )


def _srgb_to_linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92

    return ((channel + 0.055) / 1.055) ** 2.4


def _lab_f(value: float) -> float:
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    if value > epsilon:
        return value ** (1.0 / 3.0)

    return (kappa * value + 16.0) / 116.0

def build_final_report(
    region_results: dict,
    skin_tone: dict | None = None,
    excluded_allergens: list[str] | None = None,
    detected_condition_map: dict | None = None,
) -> dict:
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

    condition_map = build_condition_map_from_regions(regioes)
    recommendation_condition_map = {**(detected_condition_map or {}), **condition_map}
    response = {
        "tom_geral_fitzpatrick":  tom_geral,
        "tom_geral_hex":          bisenet_hex or tom_geral_hex,
        "fitzpatrick_source":     fitzpatrick_source,
        "subtom_predominante":    subtom_geral,
        "regioes":                regioes,
        "comparacao_tons":        comparacoes,
        "imperfeicoes":           todas_imperf,
        "recommendations":        [],
        # False when no shade was within the color-distance threshold — the
        # listed shades are the closest available, not a confident match.
        "recommendations_reliable": False,
        "recommendations_catalog_source": "openai_web_search",
        "recommendations_catalog_shades_considered": None,
        "recommendations_status": "pending",
        "recommendations_error": None,
        "skin_tone":              skin_tone,
        "condition_map":          condition_map,
        "recommendation_condition_map": recommendation_condition_map,
    }

    return apply_medical_triage(response)
