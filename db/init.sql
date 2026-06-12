CREATE TABLE IF NOT EXISTS analyses (
    id                     VARCHAR      PRIMARY KEY,
    created_at             TIMESTAMPTZ  DEFAULT now(),
    tom_geral_fitzpatrick  INTEGER,
    tom_geral_hex          VARCHAR,
    fitzpatrick_source     VARCHAR,
    subtom_predominante    VARCHAR,
    result_json            JSONB        NOT NULL
);
