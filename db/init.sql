CREATE TABLE IF NOT EXISTS analyses (
    id                     VARCHAR      PRIMARY KEY,
    user_id                UUID,
    created_at             TIMESTAMPTZ  DEFAULT now(),
    tom_geral_fitzpatrick  INTEGER,
    tom_geral_hex          VARCHAR,
    fitzpatrick_source     VARCHAR,
    subtom_predominante    VARCHAR,
    result_json            JSONB        NOT NULL
);

CREATE INDEX IF NOT EXISTS analyses_user_created_idx
    ON analyses (user_id, created_at DESC);

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
