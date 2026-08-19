#!/usr/bin/env python3
"""Download the selected public Kaggle source datasets without Kaggle CLI."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


DATASETS = {
    "sephora_website.zip": "https://www.kaggle.com/api/v1/datasets/download/raghadalharbi/all-products-available-on-sephora-website",
    "makeup_shades.zip": "https://www.kaggle.com/api/v1/datasets/download/utkarshx27/makeup-shades",
}

RAW_DIR = Path(__file__).parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in DATASETS.items():
        destination = RAW_DIR / filename
        print(f"Downloading {url}")
        urlretrieve(url, destination)
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
