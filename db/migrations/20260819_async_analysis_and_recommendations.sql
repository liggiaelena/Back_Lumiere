BEGIN;

ALTER TABLE analyses ADD COLUMN IF NOT EXISTS analysis_status VARCHAR(32);
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS analysis_stage VARCHAR(64);
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS analysis_error TEXT;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS input_image_path TEXT;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS input_content_type VARCHAR(100);
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS input_size_bytes BIGINT;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS lang VARCHAR(10);
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS excluded_allergens JSONB;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS analysis_started_at TIMESTAMPTZ;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS analysis_completed_at TIMESTAMPTZ;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS attempt_count INTEGER;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS active_recommendation_job_id VARCHAR;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

UPDATE analyses
SET analysis_status = COALESCE(analysis_status, 'ready'),
    lang = COALESCE(lang, result_json->>'lang', 'en'),
    excluded_allergens = COALESCE(excluded_allergens, '[]'::jsonb),
    attempt_count = COALESCE(attempt_count, 0),
    analysis_completed_at = COALESCE(analysis_completed_at, created_at),
    updated_at = COALESCE(updated_at, created_at, now());

ALTER TABLE analyses ALTER COLUMN analysis_status SET DEFAULT 'ready';
ALTER TABLE analyses ALTER COLUMN analysis_status SET NOT NULL;
ALTER TABLE analyses ALTER COLUMN lang SET DEFAULT 'en';
ALTER TABLE analyses ALTER COLUMN lang SET NOT NULL;
ALTER TABLE analyses ALTER COLUMN excluded_allergens SET DEFAULT '[]'::jsonb;
ALTER TABLE analyses ALTER COLUMN excluded_allergens SET NOT NULL;
ALTER TABLE analyses ALTER COLUMN attempt_count SET DEFAULT 0;
ALTER TABLE analyses ALTER COLUMN attempt_count SET NOT NULL;
ALTER TABLE analyses ALTER COLUMN updated_at SET DEFAULT now();
ALTER TABLE analyses ALTER COLUMN updated_at SET NOT NULL;

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

COMMIT;
