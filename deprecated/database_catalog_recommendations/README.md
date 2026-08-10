# Deprecated database-catalog recommendation system

This directory preserves the retired PostgreSQL/Neon product-catalog approach.
It is not imported by the active application.

Contents:

- `app/recommendations.py`: color-distance ranking over database shades.
- `app/product_service.py`: product and shade table access.
- `product_sync.py` and `product_sources.json`: retailer/brand synchronization.
- `db/init.sql`: schema snapshot before retirement.
- `tests/`: legacy catalog and ranking tests.
- `LEGACY_README.md`: documentation snapshot.

The active application now uses the OpenAI Responses API with Web search after
skin analysis. Keeping this directory makes the old implementation recoverable
without leaving it on the runtime import path.
