import json
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import DATABASE_URL, engine

logger = logging.getLogger(__name__)


def catalog_source_name() -> str:
    """Return a safe source label without exposing database credentials."""
    hostname = (urlparse(DATABASE_URL).hostname or "").casefold()
    return "neon_postgres" if hostname.endswith("neon.tech") else "postgresql"


DDL = """
CREATE TABLE IF NOT EXISTS makeup_products (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    data_source VARCHAR(255) NOT NULL,
    external_id VARCHAR(255) NOT NULL,
    brand VARCHAR(255) NOT NULL,
    name VARCHAR(500) NOT NULL,
    product_url TEXT NOT NULL,
    image_url TEXT,
    currency VARCHAR(3),
    price NUMERIC(12, 2),
    ingredients JSONB NOT NULL DEFAULT '[]'::jsonb,
    allergens JSONB NOT NULL DEFAULT '[]'::jsonb,
    available BOOLEAN NOT NULL DEFAULT TRUE,
    fetched_at TIMESTAMPTZ NOT NULL,
    UNIQUE(source, external_id)
);
CREATE TABLE IF NOT EXISTS makeup_product_shades (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL REFERENCES makeup_products(id) ON DELETE CASCADE,
    external_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    shade_code VARCHAR(100),
    shade_hex VARCHAR(7) NOT NULL,
    shade_url TEXT,
    shade_hex_source VARCHAR(50) NOT NULL DEFAULT 'official_structured_swatch',
    shade_hex_confidence NUMERIC(5, 4) NOT NULL DEFAULT 1.0,
    undertone VARCHAR(20),
    available BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE(product_id, external_id)
);
CREATE INDEX IF NOT EXISTS ix_makeup_products_available ON makeup_products(available);
CREATE INDEX IF NOT EXISTS ix_makeup_product_shades_product ON makeup_product_shades(product_id);
ALTER TABLE makeup_product_shades ADD COLUMN IF NOT EXISTS shade_hex_source VARCHAR(50) NOT NULL DEFAULT 'official_structured_swatch';
ALTER TABLE makeup_product_shades ADD COLUMN IF NOT EXISTS shade_hex_confidence NUMERIC(5, 4) NOT NULL DEFAULT 1.0;
ALTER TABLE makeup_product_shades ADD COLUMN IF NOT EXISTS shade_url TEXT;
ALTER TABLE makeup_products ADD COLUMN IF NOT EXISTS data_source VARCHAR(255) NOT NULL DEFAULT 'legacy';
"""


def ensure_product_tables() -> None:
    with engine.begin() as connection:
        for statement in DDL.split(";"):
            if statement.strip():
                connection.execute(text(statement))


def upsert_product(record: dict) -> int:
    fetched_at = record.get("fetched_at") or datetime.now(timezone.utc)
    with Session(engine) as session:
        product_id = session.execute(text("""
            INSERT INTO makeup_products
                (source, data_source, external_id, brand, name, product_url, image_url, currency,
                 price, ingredients, allergens, available, fetched_at)
            VALUES
                (:source, :data_source, :external_id, :brand, :name, :product_url, :image_url, :currency,
                 :price, CAST(:ingredients AS JSONB), CAST(:allergens AS JSONB), :available, :fetched_at)
            ON CONFLICT (source, external_id) DO UPDATE SET
                data_source = EXCLUDED.data_source, brand = EXCLUDED.brand, name = EXCLUDED.name,
                product_url = EXCLUDED.product_url, image_url = EXCLUDED.image_url,
                currency = EXCLUDED.currency, price = EXCLUDED.price,
                ingredients = EXCLUDED.ingredients, allergens = EXCLUDED.allergens,
                available = EXCLUDED.available, fetched_at = EXCLUDED.fetched_at
            RETURNING id
        """), {
            **record,
            "ingredients": json.dumps(record.get("ingredients", [])),
            "allergens": json.dumps(record.get("allergens", [])),
            "available": record.get("available", True),
            "fetched_at": fetched_at,
        }).scalar_one()

        for shade in record.get("shades", []):
            session.execute(text("""
                INSERT INTO makeup_product_shades
                    (product_id, external_id, name, shade_code, shade_hex, shade_url, shade_hex_source,
                     shade_hex_confidence, undertone, available)
                VALUES
                    (:product_id, :external_id, :name, :shade_code, :shade_hex, :shade_url, :shade_hex_source,
                     :shade_hex_confidence, :undertone, :available)
                ON CONFLICT (product_id, external_id) DO UPDATE SET
                    name = EXCLUDED.name, shade_code = EXCLUDED.shade_code,
                    shade_hex = EXCLUDED.shade_hex, shade_url = EXCLUDED.shade_url,
                    undertone = EXCLUDED.undertone,
                    shade_hex_source = EXCLUDED.shade_hex_source,
                    shade_hex_confidence = EXCLUDED.shade_hex_confidence,
                    available = EXCLUDED.available
            """), {
                "product_id": product_id,
                "external_id": str(shade.get("external_id") or shade["name"]),
                "name": shade["name"],
                "shade_code": shade.get("shade_code"),
                "shade_hex": shade["shade_hex"].lower(),
                "shade_url": shade.get("shade_url"),
                "shade_hex_source": shade.get("shade_hex_source", "official_structured_swatch"),
                "shade_hex_confidence": float(shade.get("shade_hex_confidence", 1.0)),
                "undertone": shade.get("undertone"),
                "available": shade.get("available", True),
            })
        session.commit()
    return product_id


def load_available_shades(excluded_allergens: list[str] | None = None) -> list[dict]:
    excluded = {item.strip().casefold() for item in (excluded_allergens or []) if item.strip()}
    with Session(engine) as session:
        rows = session.execute(text("""
            SELECT p.id AS product_id, p.data_source, p.brand, p.name AS product_name, p.product_url,
                   p.image_url, p.currency, p.price, p.ingredients, p.allergens, p.fetched_at,
                   s.name AS shade_name, s.shade_code, s.shade_hex, s.shade_url, s.shade_hex_source,
                   s.shade_hex_confidence, s.undertone
            FROM makeup_products p
            JOIN makeup_product_shades s ON s.product_id = p.id
            WHERE p.available = TRUE AND s.available = TRUE
              AND s.shade_hex_confidence >= 0.75
        """)).mappings().all()

    result = []
    for row in rows:
        # Allergy filtering is fail-closed: a product without a successfully
        # parsed ingredient list is not shown when the user selected exclusions.
        if excluded and not row["ingredients"]:
            continue
        product_allergens = {str(value).strip().casefold() for value in (row["allergens"] or [])}
        if excluded & product_allergens:
            continue
        result.append(dict(row))
    return result


def list_catalog_allergens() -> list[str]:
    with Session(engine) as session:
        rows = session.execute(text("""
            SELECT DISTINCT jsonb_array_elements_text(allergens) AS allergen
            FROM makeup_products
            WHERE available = TRUE
            ORDER BY allergen
        """)).scalars().all()
    return [str(value) for value in rows]
