# Makeup Product Database Builder

Builds draft concealer and primer databases from open product datasets before
anything is copied into `dev/app/recommendations.py`.

## Inputs

The selected Kaggle sources for the first pass are:

- `raghadalharbi/all-products-available-on-sephora-website`
  - CC0, 9k+ Sephora products, useful for primer products, prices, product URLs,
    ingredients, and sensitive-skin claims.
- `utkarshx27/makeup-shades`
  - CC0, swatch/shade rows with `brand`, `product`, `specific`, and `hex`, useful
    for concealer-like shade records.
- Optional enrichment: Open Beauty Facts export or API JSON from
  `world.openbeautyfacts.org`.

Other candidates reviewed:

- `nadyinky/sephora-products-and-skincare-reviews`: strong review dataset, but
  skincare-focused and less useful for concealer/primer makeup inventory.
- `shivamb/makeup-shades-dataset`: small CC0 foundation-shade dataset; useful,
  but less complete than `utkarshx27/makeup-shades` for swatch processing.
- `anamararullorti/products-catalog-dataset`: promising shade/undertone catalog,
  but skipped for now because Kaggle reports an unknown license.

You can download the selected public Kaggle files without installing Kaggle CLI:

```bash
python dev/product_database_builder/download_kaggle_sources.py
unzip -o dev/product_database_builder/data/raw/sephora_website.zip \
  -d dev/product_database_builder/data/raw/sephora_website
unzip -o dev/product_database_builder/data/raw/makeup_shades.zip \
  -d dev/product_database_builder/data/raw/makeup_shades
```

Download Open Beauty Facts ingredient references:

```bash
python dev/product_database_builder/download_open_beauty_facts.py
```

Keep raw downloads under `dev/product_database_builder/data/raw/`; this path is
ignored by git.

## Usage

```bash
python dev/product_database_builder/build_makeup_databases.py \
  --sephora-products dev/product_database_builder/data/raw/sephora_website/sephora_website_dataset.csv \
  --makeup-shades dev/product_database_builder/data/raw/makeup_shades/allShades.csv \
  --open-beauty-facts \
    dev/product_database_builder/data/raw/open_beauty_facts_concealer.json \
    dev/product_database_builder/data/raw/open_beauty_facts_primer.json
```

By default this writes:

```text
dev/product_database_builder/generated/makeup_databases.py
data-collection/makeup_products/concealers.json
data-collection/makeup_products/concealers.csv
data-collection/makeup_products/primers.json
data-collection/makeup_products/primers.csv
data-collection/makeup_products/open_beauty_facts_reference.json
data-collection/makeup_products/open_beauty_facts_reference.csv
```

The generated Python file contains `_CONCEALER_DATABASE` and `_PRIMER_DATABASE`
for quick inspection. The `data-collection/makeup_products/` JSON/CSV files are
the intended dataset artifacts to review before importing into a database.

## Notes

- Concealer shade hex values are taken from source rows when present, then
  enriched from the makeup-shades dataset by brand/product/shade matching.
- Sensitive-skin fields are conservative:
  - `True` only when source text explicitly claims the attribute.
  - `False` only when evidence contradicts it, such as fragrance ingredients.
  - `None` when unknown.
- Undertone is inferred from shade labels and codes. Review these values before
  shipping because product naming conventions vary by brand.
