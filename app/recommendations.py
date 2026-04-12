from app.color_utils import color_delta

_DATABASE = [
    # ── FENTY BEAUTY ──────────────────────────────────────────────────────────
    {"brand": "Fenty Beauty", "shade_name": "110N", "shade_code": "110N",
     "undertone": "neutro",  "shade_hex": "#f5e4d2", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "120W", "shade_code": "120W",
     "undertone": "quente",  "shade_hex": "#f2dcc6", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "130C", "shade_code": "130C",
     "undertone": "frio",    "shade_hex": "#efd6c4", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "230N", "shade_code": "230N",
     "undertone": "neutro",  "shade_hex": "#e8c9ae", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "240W", "shade_code": "240W",
     "undertone": "quente",  "shade_hex": "#e5c0a0", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "250C", "shade_code": "250C",
     "undertone": "frio",    "shade_hex": "#dbb89a", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "330N", "shade_code": "330N",
     "undertone": "neutro",  "shade_hex": "#c9976f", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "340W", "shade_code": "340W",
     "undertone": "quente",  "shade_hex": "#c48c62", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "350C", "shade_code": "350C",
     "undertone": "frio",    "shade_hex": "#be8860", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "430N", "shade_code": "430N",
     "undertone": "neutro",  "shade_hex": "#9b6840", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "440W", "shade_code": "440W",
     "undertone": "quente",  "shade_hex": "#96603a", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "450C", "shade_code": "450C",
     "undertone": "frio",    "shade_hex": "#905c3a", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "490N", "shade_code": "490N",
     "undertone": "neutro",  "shade_hex": "#5c3520", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "495W", "shade_code": "495W",
     "undertone": "quente",  "shade_hex": "#582f1a", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},
    {"brand": "Fenty Beauty", "shade_name": "498C", "shade_code": "498C",
     "undertone": "frio",    "shade_hex": "#4e2a18", "price_range": "$38–$42", "where_to_buy": "https://www.fentybeauty.com"},

    # ── MAC ───────────────────────────────────────────────────────────────────
    {"brand": "MAC", "shade_name": "NC10", "shade_code": "NC10",
     "undertone": "quente",  "shade_hex": "#f2dcc5", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "NW10", "shade_code": "NW10",
     "undertone": "frio",    "shade_hex": "#eed5c2", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "N3",   "shade_code": "N3",
     "undertone": "neutro",  "shade_hex": "#f0d9c3", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "NC20", "shade_code": "NC20",
     "undertone": "quente",  "shade_hex": "#e8c8aa", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "NW20", "shade_code": "NW20",
     "undertone": "frio",    "shade_hex": "#dfc0a0", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "N4",   "shade_code": "N4",
     "undertone": "neutro",  "shade_hex": "#e3c4a5", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "NC35", "shade_code": "NC35",
     "undertone": "quente",  "shade_hex": "#c9966a", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "NW35", "shade_code": "NW35",
     "undertone": "frio",    "shade_hex": "#c09060", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "N5",   "shade_code": "N5",
     "undertone": "neutro",  "shade_hex": "#c49365", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "NC45", "shade_code": "NC45",
     "undertone": "quente",  "shade_hex": "#97633c", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "NW45", "shade_code": "NW45",
     "undertone": "frio",    "shade_hex": "#8e5d38", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "NC50", "shade_code": "NC50",
     "undertone": "quente",  "shade_hex": "#5e3620", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "NW55", "shade_code": "NW55",
     "undertone": "frio",    "shade_hex": "#56301c", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},
    {"brand": "MAC", "shade_name": "N7",   "shade_code": "N7",
     "undertone": "neutro",  "shade_hex": "#5a3320", "price_range": "$35–$40", "where_to_buy": "https://www.maccosmetics.com"},

    # ── MAYBELLINE ────────────────────────────────────────────────────────────
    {"brand": "Maybelline", "shade_name": "110 Porcelain",    "shade_code": "110",
     "undertone": "neutro",  "shade_hex": "#f4e2d0", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "115 Ivory",        "shade_code": "115",
     "undertone": "quente",  "shade_hex": "#f1d9c2", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "120 Classic Ivory","shade_code": "120",
     "undertone": "frio",    "shade_hex": "#eed5bf", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "220 Natural Beige","shade_code": "220",
     "undertone": "neutro",  "shade_hex": "#e5c5a8", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "230 Sand Beige",   "shade_code": "230",
     "undertone": "quente",  "shade_hex": "#e2be9e", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "240 Golden Beige", "shade_code": "240",
     "undertone": "frio",    "shade_hex": "#dcb898", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "310 Sun Beige",    "shade_code": "310",
     "undertone": "neutro",  "shade_hex": "#c8906a", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "320 Warm Nude",    "shade_code": "320",
     "undertone": "quente",  "shade_hex": "#c28860", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "330 Toffee",       "shade_code": "330",
     "undertone": "frio",    "shade_hex": "#bc845c", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "360 Mocha",        "shade_code": "360",
     "undertone": "neutro",  "shade_hex": "#925840", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "370 Java",         "shade_code": "370",
     "undertone": "quente",  "shade_hex": "#8d5038", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "380 Espresso",     "shade_code": "380",
     "undertone": "frio",    "shade_hex": "#874e38", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "395 Deep Espresso","shade_code": "395",
     "undertone": "neutro",  "shade_hex": "#4e2c18", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "397 Ebony",        "shade_code": "397",
     "undertone": "quente",  "shade_hex": "#4a2814", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
    {"brand": "Maybelline", "shade_name": "399 Onyx",         "shade_code": "399",
     "undertone": "frio",    "shade_hex": "#452513", "price_range": "$10–$14", "where_to_buy": "https://www.maybelline.com"},
]

# Undertone priority when no exact-undertone match is close enough
_UNDERTONE_FALLBACK = {
    "quente": ["quente", "neutro", "frio"],
    "frio":   ["frio",   "neutro", "quente"],
    "neutro": ["neutro", "quente", "frio"],
}

# Max color-distance to consider a shade "close enough" to the skin tone
_MAX_DELTA = 60


def get_recommendations(
    fitzpatrick: int,
    undertone: str,
    skin_hex: str | None = None,
    max_per_brand: int = 1,
) -> list[dict]:
    """
    Return the best-matching shade per brand.

    When skin_hex is provided (BiSeNet median), shades are ranked by color
    distance to the actual skin tone, filtered by undertone preference.
    Falls back to Fitzpatrick-range matching when skin_hex is absent.
    """
    priority = _UNDERTONE_FALLBACK.get(undertone, ["neutro", "quente", "frio"])

    if skin_hex:
        return _match_by_color(skin_hex, priority, max_per_brand)
    else:
        return _match_by_fitzpatrick(fitzpatrick, priority, max_per_brand)


def _match_by_color(skin_hex: str, priority: list, max_per_brand: int) -> list:
    """Rank shades by color distance to skin_hex, respecting undertone priority."""
    results = []
    brands_seen: dict[str, int] = {}

    # Score every shade: primary sort = undertone priority tier, secondary = color delta
    scored = []
    for shade in _DATABASE:
        delta = color_delta(skin_hex, shade["shade_hex"])
        if delta > _MAX_DELTA:
            continue
        tier = priority.index(shade["undertone"]) if shade["undertone"] in priority else len(priority)
        scored.append((tier, delta, shade))

    scored.sort(key=lambda x: (x[0], x[1]))

    for _, _, shade in scored:
        brand = shade["brand"]
        if brands_seen.get(brand, 0) >= max_per_brand:
            continue
        results.append(_format(shade))
        brands_seen[brand] = brands_seen.get(brand, 0) + 1

    # If color filter was too strict, fall back to closest shades regardless of delta
    if not results:
        all_scored = sorted(
            _DATABASE,
            key=lambda s: (
                priority.index(s["undertone"]) if s["undertone"] in priority else len(priority),
                color_delta(skin_hex, s["shade_hex"])
            )
        )
        for shade in all_scored:
            brand = shade["brand"]
            if brands_seen.get(brand, 0) >= max_per_brand:
                continue
            results.append(_format(shade))
            brands_seen[brand] = brands_seen.get(brand, 0) + 1

    return results


def _match_by_fitzpatrick(fitzpatrick: int, priority: list, max_per_brand: int) -> list:
    """Legacy Fitzpatrick-range matching, used when no skin hex is available."""
    # Map Fitzpatrick to approximate skin hex using the same reference points as color_utils
    _FITZPATRICK_HEX = {
        1: "#f6ede4", 2: "#f3d7c0", 3: "#dba98a",
        4: "#b07d5b", 5: "#7d4e2d", 6: "#3e1f0e",
    }
    skin_hex = _FITZPATRICK_HEX.get(fitzpatrick, "#c68b6e")
    return _match_by_color(skin_hex, priority, max_per_brand)


def _format(shade: dict) -> dict:
    return {
        "brand":       shade["brand"],
        "shade_name":  shade["shade_name"],
        "shade_code":  shade["shade_code"],
        "shade_hex":   shade["shade_hex"],
        "undertone":   shade["undertone"],
        "price_range": shade["price_range"],
        "where_to_buy": shade["where_to_buy"],
    }
