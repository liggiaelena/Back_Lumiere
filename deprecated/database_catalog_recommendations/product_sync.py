"""Synchronize foundation products from official product pages.

Usage: python dev/product_sync.py --sources dev/product_sources.json
The source file contains official product URLs and, where a retailer does not
expose swatch colours as structured data, a curated `shades` list.
"""
import argparse
import hashlib
import html
import io
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.product_service import ensure_product_tables, upsert_product

KNOWN_ALLERGENS = {
    "fragrance": ("fragrance", "parfum", "aroma"),
    "limonene": ("limonene",),
    "linalool": ("linalool",),
    "citral": ("citral",),
    "geraniol": ("geraniol",),
    "eugenol": ("eugenol",),
    "latex": ("latex",),
    "lanolin": ("lanolin",),
    "propylene glycol": ("propylene glycol",),
    "phenoxyethanol": ("phenoxyethanol",),
    "methylisothiazolinone": ("methylisothiazolinone",),
    "benzyl alcohol": ("benzyl alcohol",),
    "benzyl benzoate": ("benzyl benzoate",),
    "benzyl salicylate": ("benzyl salicylate",),
    "cinnamal": ("cinnamal", "cinnamyl alcohol"),
    "coumarin": ("coumarin",),
    "isoeugenol": ("isoeugenol",),
    "oakmoss": ("evernia prunastri", "oakmoss"),
    "almond": ("prunus amygdalus dulcis", "sweet almond"),
    "soy": ("glycine soja", "soybean"),
    "wheat": ("triticum vulgare", "wheat"),
}


def _fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "LumiereCatalogBot/1.0 (+product-data-sync)"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_bytes(url: str) -> bytes:
    if url.startswith("//"):
        url = "https:" + url
    request = Request(html.unescape(url), headers={"User-Agent": "LumiereCatalogBot/1.0 (+product-data-sync)"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def _estimate_swatch_hex(image_bytes: bytes) -> tuple[str, float]:
    """Estimate a representative colour from an official swatch image.

    Near-white backgrounds, black text, and highly saturated UI pixels are
    discarded. Confidence combines usable-pixel coverage and colour cohesion.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((160, 160))
    width, height = image.size
    pixels = []
    for y in range(height // 10, height - height // 10):
        for x in range(width // 10, width - width // 10):
            r, g, b = image.getpixel((x, y))
            brightness = (r + g + b) / 3
            if not 25 <= brightness <= 242 or max(r, g, b) - min(r, g, b) < 8:
                continue
            # Broad skin/foundation gamut; rejects most blue/green UI artwork.
            if r >= g * 0.82 and g >= b * 0.72:
                pixels.append((r, g, b))
    if len(pixels) < max(20, width * height * 0.01):
        raise ValueError("swatch image has too few foundation-coloured pixels")
    rgb = tuple(int(statistics.median(channel)) for channel in zip(*pixels))
    deviations = [sum(abs(pixel[i] - rgb[i]) for i in range(3)) / 3 for pixel in pixels]
    cohesion = max(0.0, 1.0 - statistics.median(deviations) / 80.0)
    coverage = min(1.0, len(pixels) / max(1, width * height * 0.35))
    confidence = round(0.55 * cohesion + 0.45 * coverage, 4)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}", confidence


def _image_shades(document: str) -> list[dict]:
    candidates = []
    for tag in re.findall(r"<img\b[^>]*>", document, re.I | re.S):
        src_match = re.search(r'\bsrc=["\']([^"\']+)["\']', tag, re.I)
        alt_match = re.search(r'\b(?:alt|title)=["\']([^"\']+)["\']', tag, re.I)
        if not src_match or not alt_match:
            continue
        src = html.unescape(src_match.group(1))
        alt = html.unescape(alt_match.group(1)).strip()
        signal = f"{src} {tag}".casefold()
        if not any(word in signal for word in ("_sw", "swatch", "texture_swatch")):
            continue
        if any(word in alt.casefold() for word in ("arm swatch", "face chart", "shadefamily")):
            continue
        candidates.append((alt, src))

    result = []
    seen_urls = set()
    for alt, src in candidates:
        if src in seen_urls:
            continue
        seen_urls.add(src)
        try:
            shade_hex, confidence = _estimate_swatch_hex(_fetch_bytes(src))
        except Exception:
            continue
        name = re.sub(
            r"\s+(?:Face Bond|Liquid Touch|Airbrush Flawless|Foundation).*?$", "", alt,
            flags=re.I,
        ).strip(" -") or alt
        result.append({
            "name": name, "shade_code": name, "shade_hex": shade_hex,
            "undertone": _undertone(alt), "available": True,
            "shade_hex_source": "official_swatch_image_estimate",
            "shade_hex_confidence": confidence,
        })
    return result


def _json_ld(document: str) -> list[dict]:
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        document, flags=re.I | re.S,
    )
    output = []
    for block in blocks:
        try:
            value = json.loads(html.unescape(block.strip()))
            output.extend(value if isinstance(value, list) else [value])
        except (json.JSONDecodeError, TypeError):
            continue
    return [item for item in output if isinstance(item, dict)]


def _embedded_json(document: str) -> list[object]:
    """Decode JSON script blocks used by Shopify, Next.js and Demandware pages."""
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", document, flags=re.I | re.S)
    output = []
    for block in blocks:
        value = html.unescape(block.strip())
        if not value or value[0] not in "[{":
            continue
        try:
            output.append(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            continue
    return output


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _undertone(name: str) -> str:
    value = name.casefold()
    code = re.match(r"\s*([cwn])\s*\d", value)
    if code:
        return {"c": "frio", "w": "quente", "n": "neutro"}[code.group(1)]
    if any(word in value for word in ("warm", "gold", "olive", "yellow")):
        return "quente"
    if any(word in value for word in ("cool", "rose", "rosy", "pink", "red")):
        return "frio"
    return "neutro"


def _discovered_shades(document: str) -> list[dict]:
    decoded_document = html.unescape(document)
    hex_keys = ("hex", "hexCode", "hex_code", "swatchHex", "swatch_hex", "colorHex", "color_code")
    name_keys = ("shadeName", "shade_name", "displayName", "display_name", "name", "title", "label")
    found = []
    for root in _embedded_json(decoded_document):
        for item in _walk(root):
            hex_value = next((item.get(key) for key in hex_keys if item.get(key)), None)
            name = next((item.get(key) for key in name_keys if item.get(key)), None)
            if not isinstance(hex_value, str) or not isinstance(name, str):
                continue
            match = re.fullmatch(r"#?([0-9a-fA-F]{6})", hex_value.strip())
            if match:
                found.append({
                    "name": name.strip(), "shade_code": str(item.get("sku") or item.get("code") or name).strip(),
                    "shade_hex": f"#{match.group(1).lower()}", "undertone": _undertone(name),
                    "available": item.get("available", True) is not False,
                })

    # Some stores serialize the swatch name and colour in JavaScript rather
    # than valid JSON. Support both common key orders without parsing scripts.
    patterns = [
        r'["\'](?:shadeName|shade_name|name|value)["\']\s*:\s*["\']([^"\']+)["\'][^{}]{0,500}?["\'](?:hex|hexCode|swatchHex|colorHex)["\']\s*:\s*["\']#?([0-9a-fA-F]{6})',
        r'["\'](?:hex|hexCode|swatchHex|colorHex)["\']\s*:\s*["\']#?([0-9a-fA-F]{6})["\'][^{}]{0,500}?["\'](?:shadeName|shade_name|name|value)["\']\s*:\s*["\']([^"\']+)["\']',
    ]
    for index, pattern in enumerate(patterns):
        for first, second in re.findall(pattern, decoded_document, flags=re.I | re.S):
            name, hex_value = (first, second) if index == 0 else (second, first)
            found.append({"name": name.strip(), "shade_code": name.strip(),
                          "shade_hex": f"#{hex_value.lower()}", "undertone": _undertone(name)})

    # Shopify themes commonly render swatches as HTML data attributes.
    html_undertones = {
        html.unescape(name).strip().casefold(): _undertone(f"{name} {descriptor}")
        for name, descriptor in re.findall(
            r'class=["\'][^"\']*config__active-label-title[^"\']*["\'][^>]*>(.*?)</span>.*?class=["\'][^"\']*config__active-label-secondary[^"\']*["\'][^>]*>(.*?)</p>',
            document, flags=re.I | re.S,
        )
    }
    for sku, name, fragment in re.findall(
        r'data-variant-sku=["\']([^"\']*)["\'][^>]*data-option-value=["\']([^"\']+)["\'](.*?)(?=<li\b|</li>)',
        document, flags=re.I | re.S,
    ):
        color = re.search(r'background-color\s*:\s*#([0-9a-fA-F]{6})', fragment, re.I)
        if color:
            found.append({
                "name": html.unescape(name).strip(), "shade_code": sku.strip() or html.unescape(name).strip(),
                "shade_hex": f"#{color.group(1).lower()}",
                "undertone": html_undertones.get(html.unescape(name).strip().casefold(), _undertone(name)),
                "available": "data-option-coming-soon=\"true\"" not in fragment,
            })

    deduped = {}
    hex_counts = {}
    for shade in found:
        hex_counts[shade["shade_hex"]] = hex_counts.get(shade["shade_hex"], 0) + 1
    for shade in found:
        if hex_counts[shade["shade_hex"]] > 1 and shade["name"].count("-") >= 3:
            continue
        # Structured JSON carries availability/SKU and is appended first;
        # do not let the looser regex fallback overwrite that richer record.
        deduped.setdefault((shade["name"].casefold(), shade["shade_hex"]), shade)
    return list(deduped.values())


def _find_product(items: list[dict]) -> dict:
    for item in items:
        if item.get("@type") == "Product":
            return item
        graph = item.get("@graph", [])
        for child in graph if isinstance(graph, list) else []:
            if isinstance(child, dict) and child.get("@type") == "Product":
                return child
    return {}


def _ingredients(document: str, configured) -> list[str]:
    if configured:
        return [str(value).strip() for value in configured if str(value).strip()]
    for root in _json_ld(document):
        for item in _walk(root):
            if str(item.get("name", "")).strip().casefold() == "ingredients" and item.get("value"):
                value = str(item["value"]).strip()
                parts = [part.strip(" .") for part in value.split(",") if part.strip()]
                return parts if len(parts) > 2 else [value]
    panel = re.search(
        r'id=["\'][^"\']*ingredients[^"\']*Panel["\'][^>]*>.*?<section>\s*<h2>.*?</h2>\s*<p>(.*?)</p>',
        document, re.I | re.S,
    )
    if panel:
        value = re.sub(r"<[^>]+>", " ", panel.group(1))
        value = html.unescape(re.sub(r"\s+", " ", value))
        return [part.strip(" .") for part in value.split(",") if part.strip()]
    json_value = re.search(r'["\']ingredients["\']\s*:\s*"((?:\\.|[^"\\])*)"', document, re.I | re.S)
    if json_value:
        try:
            value = json.loads(f'"{json_value.group(1)}"')
            return [part.strip(" .") for part in value.split(",") if part.strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    match = re.search(r'ingredients?[^>]{0,100}>\s*([^<]{20,5000})<', document, re.I)
    if not match:
        return []
    return [part.strip(" .") for part in html.unescape(match.group(1)).split(",") if part.strip()]


def _allergens(ingredients: list[str]) -> list[str]:
    value = " ".join(ingredients).casefold()
    return sorted(name for name, aliases in KNOWN_ALLERGENS.items() if any(alias in value for alias in aliases))


def _with_shade_urls(source: dict, shades: list[dict]) -> list[dict]:
    template = source.get("shade_url_template")
    strategy = source.get("shade_url_strategy")
    if not template and not strategy:
        return shades
    result = []
    for index, shade in enumerate(shades):
        code = str(shade.get("shade_code") or "")
        number_match = re.search(r"(\d{2})$", code)
        shade_number = 100 + int(number_match.group(1)) if number_match else None
        digits = "".join(character for character in code if character.isdigit())
        shade_url = None
        if template and shade_number is not None:
            shade_url = template.format(shade_code=code, shade_number=shade_number)
        elif strategy == "mufe_hydra_standard" and len(digits) == 3:
            prefix = {"N": "061", "R": "062", "Y": "063"}.get(code[1:2].upper())
            if prefix:
                shade_url = f"https://www.makeupforever.com/us/en/face/foundation/hd-skin-hydra-glow-I000{prefix}{digits}.html"
        elif strategy == "mufe_hydra_mini" and len(digits) == 3:
            mini_digits = f"{int(digits[0]) + 4}{digits[-2:]}"
            shade_url = f"https://www.makeupforever.com/us/en/face/foundation/hd-skin-hydra-glow---mini-I000061{mini_digits}.html"
        elif strategy == "mufe_matte_velvet" and len(digits) == 3:
            shade_url = f"https://www.makeupforever.com/us/en/face/foundation/hd-skin-matte-velvet-I000064{digits}.html"
        elif strategy == "shopify_variant_sequence":
            variant_id = int(source["shade_variant_start"]) + index * int(source["shade_variant_step"])
            shade_url = f"{source['shade_variant_base_url']}?variant={variant_id}"
        result.append({**shade, "shade_url": shade.get("shade_url") or shade_url})
    return result


def _collect_one(source: dict, product_url: str) -> dict:
    document = _fetch(product_url)
    product = _find_product(_json_ld(document))
    offer = product.get("offers") or {}
    if isinstance(offer, list):
        offer = offer[0] if offer else {}
    brand = product.get("brand") or source.get("brand") or "Unknown"
    if isinstance(brand, dict):
        brand = brand.get("name", "Unknown")
    ingredients = _ingredients(document, source.get("ingredients"))
    external_id = str(source.get("external_id") or product.get("sku") or product.get("productID") or hashlib.sha256(product_url.encode()).hexdigest()[:24])
    # Prefer current retailer data; curated shades are only a fallback for
    # pages that expose price/ingredients but no machine-readable swatches.
    discovered_shades = _discovered_shades(document)
    image_shades = _image_shades(document) if not discovered_shades else []
    curated_shades = source.get("shades", [])
    shades = curated_shades if source.get("prefer_curated_shades") and curated_shades else max(
        (discovered_shades, image_shades, curated_shades), key=len
    )
    if not shades:
        raise ValueError("No shade HEX values found; add curated shades to the source configuration")
    if shades is curated_shades:
        shades = [
            {
                **shade,
                "shade_hex_source": shade.get("shade_hex_source", "curated_screenshot_estimate"),
                "shade_hex_confidence": shade.get("shade_hex_confidence", 0.8),
            }
            for shade in shades
        ]
    shades = _with_shade_urls(source, shades)

    price = source.get("price") if source.get("use_configured_price") else offer.get("price")
    currency = source.get("currency") if source.get("use_configured_price") else offer.get("priceCurrency")
    if price is None:
        price_match = re.search(r'<meta[^>]+(?:property|itemprop)=["\'](?:product:price:amount|price)["\'][^>]+content=["\']([0-9.]+)', document, re.I)
        price = price_match.group(1) if price_match else None
    if price is None:
        price_patterns = (
            r'["\']price["\']\s*:\s*["\']?([0-9]+(?:\.[0-9]{1,2})?)',
            r'(?:regular price|current price|product-price)[^$]{0,100}\$\s*([0-9]+(?:\.[0-9]{1,2})?)',
        )
        for pattern in price_patterns:
            price_match = re.search(pattern, document, re.I)
            if price_match:
                price = price_match.group(1)
                break
    if price is None:
        raise ValueError("No live price found on the official page")
    if not currency:
        currency_match = re.search(r'<meta[^>]+(?:property|itemprop)=["\'](?:product:price:currency|priceCurrency)["\'][^>]+content=["\']([A-Z]{3})', document, re.I)
        currency = currency_match.group(1) if currency_match else source.get("currency", "USD")
    return {
        "source": source.get("source", "official"),
        "data_source": urlparse(product_url).netloc.lower(),
        "external_id": external_id,
        "brand": brand,
        "name": source["name"] if source.get("use_configured_identity") else (product.get("name") or source["name"]),
        "product_url": product_url,
        "image_url": product.get("image") if isinstance(product.get("image"), str) else None,
        "currency": currency,
        "price": price,
        "ingredients": ingredients,
        "allergens": _allergens(ingredients),
        "available": "OutOfStock" not in str(offer.get("availability", "")),
        "fetched_at": datetime.now(timezone.utc),
        "shades": shades,
    }


def collect(source: dict) -> dict:
    errors = []
    product_urls = [] if source.get("force_static") else [source["url"], *source.get("fallback_urls", [])]
    for product_url in product_urls:
        try:
            return _collect_one(source, product_url)
        except Exception as exc:
            errors.append(f"{urlparse(product_url).netloc}: {exc}")
    if source.get("allow_static_fallback") and source.get("shades") and source.get("price") is not None:
        return {
            "source": source.get("source", "manual"),
            "data_source": source.get("static_data_source", "user_supplied_official_screenshot"),
            "external_id": str(source.get("external_id") or hashlib.sha256(source["url"].encode()).hexdigest()[:24]),
            "brand": source.get("brand", "Unknown"),
            "name": source["name"],
            "product_url": source["url"],
            "image_url": source.get("image_url"),
            "currency": source.get("currency", "USD"),
            "price": source["price"],
            "ingredients": source.get("ingredients", []),
            "allergens": _allergens(source.get("ingredients", [])),
            "available": True,
            "fetched_at": datetime.now(timezone.utc),
            "shades": _with_shade_urls(source, [{**shade, "shade_hex_source": "curated_screenshot_estimate",
                        "shade_hex_confidence": shade.get("shade_hex_confidence", 0.8)}
                       for shade in source["shades"]]),
        }
    raise ValueError("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--external-id", help="Synchronize only one configured external_id")
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    if args.external_id:
        sources = [source for source in sources if source.get("external_id") == args.external_id]
        if not sources:
            raise SystemExit(f"No source configured for external_id={args.external_id}")
    ensure_product_tables()
    failures = 0
    for source in sources:
        try:
            record = collect(source)
            upsert_product(record)
            print(f"synced {record['brand']} {record['name']} ({len(record['shades'])} shades)")
        except Exception as exc:
            # Upsert happens only after a complete parse, so a retailer layout
            # change cannot erase the last known-good catalogue row.
            print(f"FAILED {source.get('url')}: {exc}", file=sys.stderr)
            failures += 1
    if failures:
        raise SystemExit(f"{failures} of {len(sources)} sources failed")


if __name__ == "__main__":
    main()
