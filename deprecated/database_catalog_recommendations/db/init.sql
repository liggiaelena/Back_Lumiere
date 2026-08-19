CREATE TABLE IF NOT EXISTS analyses (
    id                     VARCHAR      PRIMARY KEY,
    created_at             TIMESTAMPTZ  DEFAULT now(),
    tom_geral_fitzpatrick  INTEGER,
    tom_geral_hex          VARCHAR,
    fitzpatrick_source     VARCHAR,
    subtom_predominante    VARCHAR,
    result_json            JSONB        NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id                      UUID         PRIMARY KEY,
    username                VARCHAR(50)  NOT NULL UNIQUE,
    email                   VARCHAR(255) NOT NULL UNIQUE,
    password_hash           VARCHAR(255) NOT NULL,
    first_name              VARCHAR(100),
    last_name               VARCHAR(100),
    age                     INTEGER,
    skin_type_self_assessed VARCHAR(50),
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now()
);

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
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
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
