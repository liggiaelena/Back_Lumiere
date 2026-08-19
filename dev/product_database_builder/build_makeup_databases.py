#!/usr/bin/env python3
"""Build concealer and primer database constants from open product data files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from pprint import pformat
from typing import Any, Iterable


DEFAULT_OUTPUT = Path(__file__).parent / "generated" / "makeup_databases.py"
DEFAULT_DATASET_DIR = Path(__file__).parents[2] / "data-collection" / "makeup_products"

CONCEALER_TERMS = ("concealer", "corrector", "anti-cernes", "cache-cernes")
PRIMER_TERMS = ("primer", "base makeup", "makeup base", "prebase")
FACE_PRIMER_CATEGORIES = {"face primer"}
CONCEALER_CATEGORIES = {"concealer", "color correct"}
NON_FACE_PRIMER_TERMS = (
    "eye primer",
    "hair primer",
    "lash primer",
    "lip primer",
    "nail primer",
    "mascara primer",
)
FRAGRANCE_TERMS = (
    "fragrance",
    "parfum",
    "perfume",
    "limonene",
    "linalool",
    "citronellol",
    "geraniol",
    "citral",
    "eugenol",
)


def main() -> None:
    args = parse_args()

    sephora_rows = load_many(args.sephora_products)
    shade_rows = load_many(args.makeup_shades)
    beauty_rows = load_many(args.open_beauty_facts)

    shade_index = build_shade_index(shade_rows)
    beauty_index = build_beauty_index(beauty_rows)
    product_index = build_product_index(sephora_rows)

    concealers = build_concealers(sephora_rows, shade_rows, shade_index, beauty_index, product_index)
    primers = build_primers(sephora_rows, beauty_index)

    write_python_database(args.output, concealers, primers)
    write_summary(args.output.with_suffix(".summary.md"), concealers, primers)
    write_dataset_files(args.dataset_dir, concealers, primers, beauty_rows)

    print(f"Wrote {args.output}")
    print(f"Wrote dataset files to {args.dataset_dir}")
    print(f"Concealers: {len(concealers)}")
    print(f"Primers: {len(primers)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build _CONCEALER_DATABASE and _PRIMER_DATABASE from CSV/JSON files."
    )
    parser.add_argument(
        "--sephora-products",
        nargs="*",
        type=Path,
        default=[],
        help="CSV/JSON/JSONL files with product, brand, category, shade, and price data.",
    )
    parser.add_argument(
        "--makeup-shades",
        nargs="*",
        type=Path,
        default=[],
        help="CSV/JSON/JSONL files with brand/product/shade hex color data.",
    )
    parser.add_argument(
        "--open-beauty-facts",
        nargs="*",
        type=Path,
        default=[],
        help="CSV/JSON/JSONL Open Beauty Facts product data for ingredients and claims.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Python output file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"CSV/JSON dataset output directory. Default: {DEFAULT_DATASET_DIR}",
    )
    return parser.parse_args()


def load_many(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_rows(path))
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [normalize_row(row) for row in csv.DictReader(handle)]

    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(normalize_row(json.loads(line)))
        return rows

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("products") or data.get("items") or data.get("data") or [data]
        if not isinstance(data, list):
            raise ValueError(f"Unsupported JSON shape in {path}")
        return [normalize_row(row) for row in data if isinstance(row, dict)]

    raise ValueError(f"Unsupported input format: {path}")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {normalize_key(key): value for key, value in row.items()}


def normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def build_shade_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    index: dict[tuple[str, str, str], str] = {}
    for row in rows:
        brand = clean_text(first_value(row, "brand", "brand_name", "brand_names"))
        product = clean_text(first_value(row, "product", "product_name", "name"))
        shade = clean_text(first_value(row, "shade", "shade_name", "specific", "colour", "color"))
        hex_value = clean_hex(first_value(row, "hex", "shade_hex", "color_hex", "colour_hex"))

        if not brand or not shade or not hex_value:
            continue

        index[(key_text(brand), key_text(product), key_text(shade))] = hex_value
        index[(key_text(brand), "", key_text(shade))] = hex_value
    return index


def build_beauty_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        brand = clean_text(first_value(row, "brand", "brands", "brand_name"))
        product = clean_text(first_value(row, "product_name", "generic_name", "name"))
        if brand and product:
            for brand_part in split_brands(brand):
                index[(key_text(brand_part), key_text(product))] = row
    return index


def build_product_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        brand = clean_text(first_value(row, "brand", "brands", "brand_name"))
        product = clean_text(first_value(row, "product_name", "product", "name"))
        if brand and product:
            index[(key_text(brand), key_text(product))] = row
    return index


def build_concealers(
    rows: list[dict[str, Any]],
    shade_rows: list[dict[str, Any]],
    shade_index: dict[tuple[str, str, str], str],
    beauty_index: dict[tuple[str, str], dict[str, Any]],
    product_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for row in rows:
        if not is_concealer_product(row):
            continue

        brand = clean_text(first_value(row, "brand", "brand_name", "brands"))
        product = clean_text(first_value(row, "product_name", "product", "name"))
        shade_name = clean_text(
            first_value(row, "shade_name", "shade", "variation_value", "color", "colour")
        )
        if not brand or not shade_name:
            continue

        shade_code = infer_shade_code(shade_name)
        dedupe_key = (key_text(brand), key_text(product), key_text(shade_name))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        beauty_row = find_product(beauty_index, brand, product)
        shade_hex = clean_hex(first_value(row, "shade_hex", "hex", "color_hex", "colour_hex"))
        if not shade_hex:
            shade_hex = find_shade_hex(shade_index, brand, product, shade_name)
        if not shade_hex:
            continue

        record = {
            "brand": brand,
            "shade_name": shade_name,
            "shade_code": shade_code,
            "undertone": infer_undertone(shade_name),
            "shade_hex": shade_hex,
            "finish": infer_finish(row),
            "price_range": price_range(first_value(row, "price", "price_usd", "list_price")),
            "ingredients": best_ingredients(row, beauty_row),
            "ingredients_source": ingredients_source(row, beauty_row),
            "fragrance_free": infer_fragrance_free(row, beauty_row),
            "hypoallergenic": infer_claim(row, beauty_row, ("hypoallergenic", "hypo allergenic")),
            "derm_tested": infer_claim(
                row,
                beauty_row,
                ("dermatologist tested", "dermatologically tested", "derm tested"),
            ),
            "where_to_buy": infer_url(row, brand),
            "open_beauty_facts_url": clean_text(first_value(beauty_row, "url")) or None,
            "source": "sephora/open-dataset",
        }
        records.append(record)

    for row in shade_rows:
        if not is_concealer_product(row):
            continue

        brand = clean_text(first_value(row, "brand", "brand_name", "brands"))
        product = clean_text(first_value(row, "product_name", "product", "name"))
        shade_name = clean_text(
            first_value(row, "shade_name", "shade", "specific", "color", "colour", "description")
        )
        shade_hex = clean_hex(first_value(row, "shade_hex", "hex", "color_hex", "colour_hex"))
        if not brand or not product or not shade_name or not shade_hex:
            continue

        dedupe_key = (key_text(brand), key_text(product), key_text(shade_name))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        product_row = find_product(product_index, brand, product)
        beauty_row = find_product(beauty_index, brand, product)
        source_row = product_row or row

        records.append(
            {
                "brand": clean_brand(brand),
                "shade_name": shade_name,
                "shade_code": infer_shade_code(shade_name),
                "undertone": infer_undertone(" ".join([shade_name, clean_text(row.get("description"))])),
                "shade_hex": shade_hex,
                "finish": infer_finish(source_row),
                "price_range": price_range(first_value(source_row, "price", "price_usd", "list_price")),
                "ingredients": best_ingredients(source_row, beauty_row),
                "ingredients_source": ingredients_source(source_row, beauty_row),
                "fragrance_free": infer_fragrance_free(source_row, beauty_row),
                "hypoallergenic": infer_claim(source_row, beauty_row, ("hypoallergenic", "hypo allergenic")),
                "derm_tested": infer_claim(
                    source_row,
                    beauty_row,
                    ("dermatologist tested", "dermatologically tested", "derm tested"),
                ),
                "where_to_buy": infer_url(source_row, brand),
                "open_beauty_facts_url": clean_text(first_value(beauty_row, "url")) or None,
                "source": "kaggle:utkarshx27/makeup-shades",
            }
        )

    return sorted(records, key=lambda item: (item["brand"], item["shade_name"]))


def build_primers(
    rows: list[dict[str, Any]],
    beauty_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        if not is_face_primer_product(row):
            continue

        brand = clean_text(first_value(row, "brand", "brand_name", "brands"))
        product = clean_text(first_value(row, "product_name", "product", "name"))
        if not brand or not product:
            continue

        dedupe_key = (key_text(brand), key_text(product))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        beauty_row = find_product(beauty_index, brand, product)
        records.append(
            {
                "brand": brand,
                "product_name": product,
                "finish": infer_finish(row),
                "skin_type": infer_skin_types(row),
                "price_range": price_range(first_value(row, "price", "price_usd", "list_price")),
                "ingredients": best_ingredients(row, beauty_row),
                "ingredients_source": ingredients_source(row, beauty_row),
                "fragrance_free": infer_fragrance_free(row, beauty_row),
                "hypoallergenic": infer_claim(row, beauty_row, ("hypoallergenic", "hypo allergenic")),
                "derm_tested": infer_claim(
                    row,
                    beauty_row,
                    ("dermatologist tested", "dermatologically tested", "derm tested"),
                ),
                "where_to_buy": infer_url(row, brand),
                "open_beauty_facts_url": clean_text(first_value(beauty_row, "url")) or None,
                "source": "sephora/open-dataset",
            }
        )

    return sorted(records, key=lambda item: (item["brand"], item["product_name"]))


def row_matches(row: dict[str, Any], terms: tuple[str, ...]) -> bool:
    haystack = " ".join(
        str(first_value(row, key) or "")
        for key in (
            "product_name",
            "product",
            "name",
            "category",
            "primary_category",
            "secondary_category",
            "tertiary_category",
            "product_type",
            "description",
            "details",
            "ingredients",
            "marketingflags_content",
        )
    ).lower()
    return any(term in haystack for term in terms)


def is_concealer_product(row: dict[str, Any]) -> bool:
    category = key_text(first_value(row, "category", "primary_category", "secondary_category") or "")
    text = searchable_text(row)
    if category in CONCEALER_CATEGORIES:
        return True
    if not any(term in text for term in CONCEALER_TERMS):
        return False
    return not any(term in text for term in ("brush", "sponge", "applicator"))


def is_face_primer_product(row: dict[str, Any]) -> bool:
    category = key_text(first_value(row, "category", "primary_category", "secondary_category") or "")
    text = searchable_text(row)
    if category not in FACE_PRIMER_CATEGORIES:
        return False
    if not any(term in text for term in PRIMER_TERMS):
        return False
    if any(term in text for term in NON_FACE_PRIMER_TERMS):
        return False
    return True


def first_value(row: dict[str, Any] | None, *keys: str) -> Any:
    if not row:
        return None
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_brand(value: Any) -> str:
    brand = clean_text(value)
    brand = re.sub(r"\s+by\s+Rihanna$", "", brand, flags=re.IGNORECASE)
    return brand


def split_brands(value: Any) -> list[str]:
    brands = [clean_text(part) for part in re.split(r"[,;/|]", clean_text(value))]
    return [brand for brand in brands if brand]


def key_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def clean_hex(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"#?[0-9a-fA-F]{6}", str(value))
    if not match:
        return None
    hex_value = match.group(0)
    if not hex_value.startswith("#"):
        hex_value = f"#{hex_value}"
    return hex_value.lower()


def find_shade_hex(
    shade_index: dict[tuple[str, str, str], str],
    brand: str,
    product: str,
    shade_name: str,
) -> str | None:
    brand_key = key_text(brand)
    product_key = key_text(product)
    shade_key = key_text(shade_name)
    return (
        shade_index.get((brand_key, product_key, shade_key))
        or shade_index.get((brand_key, "", shade_key))
        or find_fuzzy_shade_hex(shade_index, brand_key, product_key, shade_key)
    )


def find_fuzzy_shade_hex(
    shade_index: dict[tuple[str, str, str], str],
    brand_key: str,
    product_key: str,
    shade_key: str,
) -> str | None:
    for (brand, product, shade), hex_value in shade_index.items():
        if brand != brand_key:
            continue
        if product_key and product and product_key not in product and product not in product_key:
            continue
        if shade_key and (shade_key in shade or shade in shade_key):
            return hex_value
    return None


def find_product(
    beauty_index: dict[tuple[str, str], dict[str, Any]],
    brand: str,
    product: str,
) -> dict[str, Any] | None:
    brand_key = key_text(brand)
    product_key = key_text(product)
    if not brand_key or not product_key:
        return None
    direct = beauty_index.get((brand_key, product_key))
    if direct:
        return direct
    for (candidate_brand, candidate_product), row in beauty_index.items():
        if candidate_brand == brand_key and (
            product_key in candidate_product or candidate_product in product_key
        ):
            return row
        if candidate_brand in product_key and (
            product_key in candidate_product or candidate_product in product_key
        ):
            return row
    return None


def infer_shade_code(shade_name: str) -> str:
    code = re.match(r"^[A-Za-z]*\d+[A-Za-z]*", shade_name.strip())
    if code:
        return code.group(0).upper()
    first_token = shade_name.split()[0] if shade_name.split() else shade_name
    return re.sub(r"[^A-Za-z0-9]+", "", first_token).upper() or "NA"


def infer_undertone(shade_name: str) -> str:
    text = shade_name.lower()
    if re.search(r"\b(c|cool|rose|rosy|pink|red)\b", text):
        return "frio"
    if re.search(r"\b(w|warm|gold|golden|honey|olive|yellow)\b", text):
        return "quente"
    if re.search(r"\b(n|neutral|beige|natural)\b", text):
        return "neutro"
    return "neutro"


def infer_finish(row: dict[str, Any]) -> str:
    text = searchable_text(row)
    if any(term in text for term in ("hydrating", "moisturizing", "moisturising")):
        return "hydrating"
    if any(term in text for term in ("radiant", "glow", "luminous", "dewy")):
        return "dewy"
    if any(term in text for term in ("matte", "oil control", "pore", "blur")):
        return "matte"
    if any(term in text for term in ("full coverage", "full-coverage")):
        return "full_coverage"
    return "natural"


def infer_skin_types(row: dict[str, Any]) -> list[str]:
    text = searchable_text(row)
    skin_types = []
    if any(term in text for term in ("oily", "oil control", "mattifying")):
        skin_types.append("oleoso")
    if any(term in text for term in ("combination", "combo")):
        skin_types.append("misto")
    if any(term in text for term in ("dry", "hydrating", "moisturizing", "moisturising")):
        skin_types.append("seco")
    if "sensitive" in text:
        skin_types.append("sensivel")
    if "normal" in text:
        skin_types.append("normal")
    return skin_types or ["normal", "misto"]


def infer_fragrance_free(
    product_row: dict[str, Any],
    beauty_row: dict[str, Any] | None,
) -> bool | None:
    text = searchable_text(product_row, beauty_row)
    if any(term in text for term in ("fragrance-free", "fragrance free", "unscented", "sans parfum")):
        return True

    ingredients = " ".join(
        str(first_value(row, "ingredients_text", "ingredients", "ingredients_text_en") or "")
        for row in (product_row, beauty_row)
        if row
    ).lower()
    if any(term in ingredients for term in FRAGRANCE_TERMS):
        return False
    return None


def best_ingredients(
    product_row: dict[str, Any],
    beauty_row: dict[str, Any] | None,
) -> str | None:
    beauty_ingredients = clean_ingredients(
        first_value(beauty_row, "ingredients_text_en", "ingredients_text", "ingredients")
    )
    if beauty_ingredients:
        return beauty_ingredients
    return clean_ingredients(first_value(product_row, "ingredients", "ingredients_text"))


def ingredients_source(
    product_row: dict[str, Any],
    beauty_row: dict[str, Any] | None,
) -> str | None:
    if clean_ingredients(first_value(beauty_row, "ingredients_text_en", "ingredients_text", "ingredients")):
        return "open_beauty_facts"
    if clean_ingredients(first_value(product_row, "ingredients", "ingredients_text")):
        return "sephora"
    return None


def clean_ingredients(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    lowered = text.lower()
    placeholders = (
        "kommer senere",
        "coming soon",
        "not available",
        "unknown",
        "jshsg",
        "n/a",
    )
    if lowered in placeholders:
        return None
    if len(text) < 12 and "," not in text:
        return None
    return text


def infer_claim(
    product_row: dict[str, Any],
    beauty_row: dict[str, Any] | None,
    positive_terms: tuple[str, ...],
) -> bool | None:
    text = searchable_text(product_row, beauty_row)
    if any(term in text for term in positive_terms):
        return True
    return None


def searchable_text(*rows: dict[str, Any] | None) -> str:
    values: list[str] = []
    for row in rows:
        if not row:
            continue
        for value in row.values():
            if isinstance(value, (str, int, float, bool)):
                values.append(str(value))
            elif isinstance(value, list):
                values.extend(str(item) for item in value)
    return " ".join(values).lower()


def price_range(value: Any) -> str:
    if value in (None, ""):
        return "unknown"
    text = str(value)
    match = re.search(r"\d+(?:\.\d+)?", text.replace(",", "."))
    if not match:
        return clean_text(value) or "unknown"
    price = float(match.group(0))
    lower = int(price // 5 * 5)
    upper = lower + 4
    return f"${lower}-${upper}"


def infer_url(row: dict[str, Any], brand: str) -> str:
    url = clean_text(first_value(row, "url", "product_url", "link", "product_link"))
    if url.startswith("http://") or url.startswith("https://"):
        return url
    brand_domains = {
        "fenty beauty": "https://www.fentybeauty.com",
        "mac": "https://www.maccosmetics.com",
        "maybelline": "https://www.maybelline.com",
        "the ordinary": "https://theordinary.com",
    }
    return brand_domains.get(key_text(brand), "https://www.sephora.com")


def write_python_database(
    output: Path,
    concealers: list[dict[str, Any]],
    primers: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# Generated by dev/product_database_builder/build_makeup_databases.py\n"
        "# Review source coverage and inferred fields before importing into app code.\n\n"
        f"_CONCEALER_DATABASE = {pformat(concealers, width=100, sort_dicts=False)}\n\n"
        f"_PRIMER_DATABASE = {pformat(primers, width=100, sort_dicts=False)}\n"
    )
    output.write_text(text, encoding="utf-8")


def write_summary(
    output: Path,
    concealers: list[dict[str, Any]],
    primers: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# Makeup Database Build Summary",
            "",
            "## Sources",
            "",
            "- Kaggle: `raghadalharbi/all-products-available-on-sephora-website`",
            "- Kaggle: `utkarshx27/makeup-shades`",
            "",
            "## Output Counts",
            "",
            f"- Concealer shade records: {len(concealers)}",
            f"- Face primer product records: {len(primers)}",
            "",
            "## Caveats",
            "",
            "- Concealer shade rows come from swatch data for products whose names mention concealer/corrector.",
            "- Sephora product rows are used to enrich price, URL, ingredients, and sensitive-skin flags when product matching succeeds.",
            "- Sensitive-skin fields remain `None` unless the source text gives explicit evidence.",
            "",
        ]
    )
    output.write_text(text, encoding="utf-8")


def write_dataset_files(
    dataset_dir: Path,
    concealers: list[dict[str, Any]],
    primers: list[dict[str, Any]],
    open_beauty_facts_rows: list[dict[str, Any]],
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    open_beauty_facts_reference = build_open_beauty_facts_reference(open_beauty_facts_rows)

    (dataset_dir / "concealers.json").write_text(
        json.dumps(concealers, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "primers.json").write_text(
        json.dumps(primers, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(dataset_dir / "concealers.csv", concealers)
    write_csv(dataset_dir / "primers.csv", primers)
    (dataset_dir / "open_beauty_facts_reference.json").write_text(
        json.dumps(open_beauty_facts_reference, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(dataset_dir / "open_beauty_facts_reference.csv", open_beauty_facts_reference)
    write_dataset_readme(
        dataset_dir,
        len(concealers),
        len(primers),
        len(open_beauty_facts_reference),
    )


def build_open_beauty_facts_reference(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    seen_codes = set()
    for row in rows:
        if not is_open_beauty_facts_makeup_reference(row):
            continue

        ingredients = clean_ingredients(
            first_value(row, "ingredients_text_en", "ingredients_text", "ingredients")
        )
        if not ingredients:
            continue

        code = clean_text(first_value(row, "code", "id", "_id"))
        if code and code in seen_codes:
            continue
        if code:
            seen_codes.add(code)

        record = {
            "code": code or None,
            "brand": clean_text(first_value(row, "brands", "brand", "brand_name")) or None,
            "product_name": clean_text(first_value(row, "product_name", "generic_name", "name")) or None,
            "categories": clean_text(first_value(row, "categories")) or None,
            "ingredients": ingredients,
            "ingredients_source": "open_beauty_facts",
            "fragrance_free": infer_fragrance_free({}, row),
            "hypoallergenic": infer_claim({}, row, ("hypoallergenic", "hypo allergenic")),
            "derm_tested": infer_claim(
                {},
                row,
                ("dermatologist tested", "dermatologically tested", "derm tested"),
            ),
            "open_beauty_facts_url": clean_text(first_value(row, "url")) or None,
            "source": "open_beauty_facts",
        }
        records.append(record)
    return sorted(records, key=lambda item: (item["brand"] or "", item["product_name"] or ""))


def is_open_beauty_facts_makeup_reference(row: dict[str, Any]) -> bool:
    text = searchable_text(row)
    if any(term in text for term in ("eye primer", "lash primer", "mascara primer", "hair", "root blur", "keratina")):
        return False
    return any(term in text for term in ("concealer", "corrector", "face primer", "make-up base", "pre-make-up base"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def write_dataset_readme(
    dataset_dir: Path,
    concealer_count: int,
    primer_count: int,
    open_beauty_facts_count: int,
) -> None:
    text = "\n".join(
        [
            "# Makeup Product Datasets",
            "",
            "Generated by `dev/product_database_builder/build_makeup_databases.py`.",
            "",
            "## Files",
            "",
            "- `concealers.json` / `concealers.csv`: concealer-like shade records with hex colors.",
            "- `primers.json` / `primers.csv`: Sephora face-primer product records.",
            "- `open_beauty_facts_reference.json` / `open_beauty_facts_reference.csv`: second-source ingredient records from Open Beauty Facts.",
            "",
            "## Sources",
            "",
            "- Kaggle: `raghadalharbi/all-products-available-on-sephora-website`",
            "- Kaggle: `utkarshx27/makeup-shades`",
            "- Open Beauty Facts: `world.openbeautyfacts.org` search results for concealer and primer.",
            "",
            "## Counts",
            "",
            f"- Concealers: {concealer_count}",
            f"- Primers: {primer_count}",
            f"- Open Beauty Facts ingredient references: {open_beauty_facts_count}",
            "",
            "## Import Notes",
            "",
            "Keep this dataset as the reviewable source of truth before creating database seed SQL.",
            "Sensitive-skin fields are `true`, `false`, or `null` depending on source evidence.",
            "Main dataset rows use `ingredients_source` to show whether ingredients came from Sephora or Open Beauty Facts.",
            "The Open Beauty Facts reference file keeps filtered face-makeup ingredient records even when exact product matching was not possible.",
            "",
        ]
    )
    (dataset_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
