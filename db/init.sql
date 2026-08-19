CREATE TABLE IF NOT EXISTS analyses (
    id                     VARCHAR      PRIMARY KEY,
    user_id                UUID,
    created_at             TIMESTAMPTZ  DEFAULT now(),
    tom_geral_fitzpatrick  INTEGER,
    tom_geral_hex          VARCHAR,
    fitzpatrick_source     VARCHAR,
    subtom_predominante    VARCHAR,
    result_json            JSONB        NOT NULL,
    analysis_status        VARCHAR(32)  NOT NULL DEFAULT 'ready',
    analysis_stage         VARCHAR(64),
    analysis_error         TEXT,
    input_image_path       TEXT,
    input_content_type     VARCHAR(100),
    input_size_bytes       BIGINT,
    lang                   VARCHAR(10)  NOT NULL DEFAULT 'en',
    excluded_allergens     JSONB        NOT NULL DEFAULT '[]'::jsonb,
    analysis_started_at    TIMESTAMPTZ,
    analysis_completed_at  TIMESTAMPTZ,
    heartbeat_at           TIMESTAMPTZ,
    attempt_count          INTEGER      NOT NULL DEFAULT 0,
    active_recommendation_job_id VARCHAR,
    updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS analyses_user_created_idx
    ON analyses (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS analyses_status_created_idx
    ON analyses (analysis_status, created_at);

CREATE TABLE IF NOT EXISTS recommendation_jobs (
    id                  VARCHAR      PRIMARY KEY,
    analysis_id         VARCHAR      NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    status              VARCHAR(32)  NOT NULL DEFAULT 'queued',
    excluded_allergens  JSONB        NOT NULL DEFAULT '[]'::jsonb,
    exclusions_hash     VARCHAR(64)  NOT NULL,
    lang                VARCHAR(10)  NOT NULL DEFAULT 'en',
    force_fallback      BOOLEAN      NOT NULL DEFAULT false,
    result_json         JSONB,
    strategy            VARCHAR(64),
    model               VARCHAR,
    fallback_used       BOOLEAN      NOT NULL DEFAULT false,
    primary_error       TEXT,
    last_error          TEXT,
    attempt_count       INTEGER      NOT NULL DEFAULT 0,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    heartbeat_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (analysis_id, exclusions_hash, lang)
);

CREATE INDEX IF NOT EXISTS recommendation_jobs_status_created_idx
    ON recommendation_jobs (status, created_at);

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
