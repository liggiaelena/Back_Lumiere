#!/usr/bin/env python3
"""Download Open Beauty Facts search results for ingredient enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


RAW_DIR = Path(__file__).parent / "data" / "raw"
BASE_URL = "https://world.openbeautyfacts.org/cgi/search.pl"
FIELDS = (
    "code,brands,product_name,generic_name,categories,ingredients_text,"
    "ingredients_text_en,labels,labels_tags,url"
)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for term in ("concealer", "primer"):
        params = {
            "search_terms": term,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 100,
            "fields": FIELDS,
        }
        url = f"{BASE_URL}?{urlencode(params)}"
        output = RAW_DIR / f"open_beauty_facts_{term}.json"
        print(f"Downloading {url}")
        with urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {output} ({len(data.get('products', []))} products)")


if __name__ == "__main__":
    main()
