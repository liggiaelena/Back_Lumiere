import hashlib
import json
import logging
import secrets
from pathlib import Path

from psycopg2.extras import Json as PsycopgJson
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import engine

logger = logging.getLogger(__name__)


class AnalysisQueueFullError(RuntimeError):
    pass


def _analysis_error_code(error: str | None) -> str | None:
    """Return a stable client-facing code for known analysis failures."""
    if not error:
        return None
    normalized = error.casefold()
    known_errors = (
        ("no face detected", "no_face_detected"),
        ("multiple faces detected", "multiple_faces_detected"),
        ("too dark", "face_too_dark"),
        ("too bright", "face_overexposed"),
        ("overexposed", "face_overexposed"),
        ("bounding box is invalid", "invalid_face_region"),
        ("did not detect a usable face/skin region", "invalid_face_region"),
        ("face regions are incomplete", "incomplete_face_regions"),
        ("could not encode the detected face image", "face_encoding_failed"),
        ("segformer", "analysis_model_failed"),
        ("checkpoint", "analysis_model_failed"),
        ("model is missing", "analysis_model_failed"),
    )
    for fragment, code in known_errors:
        if fragment in normalized:
            return code
    return "analysis_failed"


def ensure_analysis_job_schema() -> None:
    migration = Path(__file__).resolve().parents[2] / "db" / "migrations" / "20260819_async_analysis_and_recommendations.sql"
    sql = migration.read_text(encoding="utf-8").strip()
    if sql.startswith("BEGIN;"):
        sql = sql[len("BEGIN;"):].strip()
    if sql.endswith("COMMIT;"):
        sql = sql[:-len("COMMIT;")].strip()
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def ensure_analysis_ownership() -> None:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE analyses ADD COLUMN IF NOT EXISTS user_id UUID"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS analyses_user_created_idx ON analyses (user_id, created_at DESC)"))


def _normalise_allergens(values):
    return sorted({str(value).strip().casefold() for value in (values or []) if str(value).strip()})


def create_analysis_job(*, user_id, image_path, content_type, size_bytes, lang, excluded_allergens, max_unfinished):
    analysis_id = "ana_" + secrets.token_hex(16)
    allergens = _normalise_allergens(excluded_allergens)
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(8665, 20260819)"))
        unfinished = conn.execute(text(
            "SELECT count(*) FROM analyses WHERE analysis_status IN ('uploading','queued','predicting_local','analyzing_regions')"
        )).scalar_one()
        if unfinished >= max_unfinished:
            raise AnalysisQueueFullError("Analysis queue is full")
        conn.execute(text("""
            INSERT INTO analyses (
                id, user_id, result_json, analysis_status, analysis_stage,
                input_image_path, input_content_type, input_size_bytes, lang,
                excluded_allergens, updated_at
            ) VALUES (
                :id, CAST(:user_id AS UUID), '{}'::jsonb, 'uploading', 'uploading',
                :path, :content_type, :size_bytes, :lang, :allergens, now()
            )
        """), {"id": analysis_id, "user_id": user_id, "path": image_path,
                 "content_type": content_type, "size_bytes": size_bytes, "lang": lang,
                 "allergens": PsycopgJson(allergens)})
        position = conn.execute(text("""
            SELECT count(*) + 1 FROM analyses WHERE analysis_status='queued'
        """), {"id": analysis_id}).scalar_one()
    return {"id": analysis_id, "analysis_status": "queued", "analysis_stage": "queued", "queue_position": position}


def activate_analysis_job(analysis_id):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE analyses SET analysis_status='queued',analysis_stage='queued',updated_at=now()
            WHERE id=:id AND analysis_status='uploading'
        """), {"id": analysis_id})


def claim_analysis_job():
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT id, input_image_path, lang, excluded_allergens
            FROM analyses WHERE analysis_status='queued' ORDER BY created_at
            FOR UPDATE SKIP LOCKED LIMIT 1
        """)).mappings().one_or_none()
        if row is None:
            return None
        conn.execute(text("""
            UPDATE analyses SET analysis_status='predicting_local', analysis_stage='predicting_local',
                analysis_started_at=COALESCE(analysis_started_at,now()), heartbeat_at=now(),
                attempt_count=attempt_count+1, analysis_error=NULL, updated_at=now()
            WHERE id=:id AND analysis_status='queued'
        """), {"id": row["id"]})
        return dict(row)


def set_analysis_stage(analysis_id, stage):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE analyses SET analysis_status=:stage, analysis_stage=:stage,
                heartbeat_at=now(), updated_at=now()
            WHERE id=:id AND analysis_status IN ('predicting_local','analyzing_regions')
        """), {"id": analysis_id, "stage": stage})


def complete_analysis_job(analysis_id, result):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE analyses SET result_json=:result, analysis_status='ready', analysis_stage='ready',
                analysis_error=NULL, tom_geral_fitzpatrick=:fitzpatrick, tom_geral_hex=:hex,
                fitzpatrick_source=:source, subtom_predominante=:subtom,
                analysis_completed_at=now(), heartbeat_at=now(), updated_at=now()
            WHERE id=:id AND analysis_status IN ('predicting_local','analyzing_regions')
        """), {"id": analysis_id, "result": PsycopgJson(result),
                 "fitzpatrick": result.get("tom_geral_fitzpatrick"), "hex": result.get("tom_geral_hex"),
                 "source": result.get("fitzpatrick_source"), "subtom": result.get("subtom_predominante")})


def mark_analysis_failed(analysis_id, error):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE analyses SET analysis_status='failed', analysis_stage='failed', analysis_error=:error,
                heartbeat_at=now(), updated_at=now()
            WHERE id=:id AND analysis_status NOT IN ('ready','cancelled','expired')
        """), {"id": analysis_id, "error": str(error)[:2000]})


def save_analysis(result, user_id=None):
    analysis_id = "ana_" + secrets.token_hex(16)
    with Session(engine) as session:
        session.execute(text("""
            INSERT INTO analyses (id,user_id,tom_geral_fitzpatrick,tom_geral_hex,
                fitzpatrick_source,subtom_predominante,result_json,analysis_status,
                analysis_stage,analysis_completed_at)
            VALUES (:id,CAST(:user_id AS UUID),:fitzpatrick,:hex,:source,:subtom,
                :result_json,'ready','ready',now())
        """), {"id": analysis_id, "user_id": user_id,
                 "fitzpatrick": result.get("tom_geral_fitzpatrick"), "hex": result.get("tom_geral_hex"),
                 "source": result.get("fitzpatrick_source"), "subtom": result.get("subtom_predominante"),
                 "result_json": PsycopgJson(result)})
        session.commit()
    return analysis_id


def list_user_analyses(user_id):
    with Session(engine) as session:
        rows = session.execute(text("""
            SELECT a.id,a.created_at,a.tom_geral_fitzpatrick,a.tom_geral_hex,
                a.subtom_predominante,a.analysis_status,
                COALESCE(r.status,'not_requested') AS recommendation_status
            FROM analyses a LEFT JOIN recommendation_jobs r ON r.id=a.active_recommendation_job_id
            WHERE a.user_id=CAST(:user_id AS UUID) ORDER BY a.created_at DESC
        """), {"user_id": user_id}).mappings().all()
    return [{**dict(row), "created_at": row.created_at.isoformat()} for row in rows]


def get_analysis_status(analysis_id, user_id=None):
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id,analysis_status,analysis_stage,analysis_error,created_at,active_recommendation_job_id
            FROM analyses WHERE id=:id AND (user_id IS NULL OR user_id=CAST(:user_id AS UUID))
        """), {"id": analysis_id, "user_id": user_id}).mappings().one_or_none()
        if row is None:
            return None
        rec = None
        if row["active_recommendation_job_id"]:
            rec = conn.execute(text("SELECT id,status,last_error FROM recommendation_jobs WHERE id=:id"),
                               {"id": row["active_recommendation_job_id"]}).mappings().one_or_none()
        position = None
        if row["analysis_status"] == "queued":
            position = conn.execute(text("""
                SELECT count(*) FROM analyses WHERE analysis_status='queued' AND created_at<=:created
            """), {"created": row["created_at"]}).scalar_one()
    intervals = {"queued": 2, "predicting_local": 2, "analyzing_regions": 3}
    analysis_error = row["analysis_error"]
    error_code = _analysis_error_code(analysis_error)
    return {"id": row["id"], "analysis_status": row["analysis_status"],
            "analysis_stage": row["analysis_stage"], "analysis_error": analysis_error,
            # `analysis_error` is retained for existing clients.  The generic
            # fields make terminal job failures consistent with recommendation
            # job status responses and easier for new clients to consume.
            "error": analysis_error, "message": analysis_error, "error_code": error_code,
            "queue_position": position, "recommendation_job_id": rec["id"] if rec else None,
            "recommendation_status": rec["status"] if rec else "not_requested",
            "recommendation_error": rec["last_error"] if rec else None,
            "poll_after_seconds": intervals.get(row["analysis_status"])}


def get_analysis(analysis_id, user_id=None):
    with Session(engine) as session:
        row = session.execute(text("""
            SELECT id,created_at,result_json,analysis_status,analysis_stage,analysis_error,
                active_recommendation_job_id FROM analyses
            WHERE id=:id AND (user_id IS NULL OR user_id=CAST(:user_id AS UUID))
        """), {"id": analysis_id, "user_id": user_id}).mappings().one_or_none()
        if row is None:
            return None
        data = dict(row["result_json"] or {})
        data.update({"id": row["id"], "created_at": row["created_at"].isoformat(),
                     "analysis_status": row["analysis_status"], "analysis_stage": row["analysis_stage"],
                     "analysis_error": row["analysis_error"]})
        if row["active_recommendation_job_id"]:
            rec = session.execute(text("SELECT * FROM recommendation_jobs WHERE id=:id"),
                                  {"id": row["active_recommendation_job_id"]}).mappings().one_or_none()
            if rec:
                data.update(recommendation_response(dict(rec)))
        return data


def create_or_get_recommendation_job(analysis_id, excluded_allergens, lang, force_fallback=False):
    allergens = _normalise_allergens(excluded_allergens)
    digest = hashlib.sha256(json.dumps(allergens, separators=(",", ":")).encode()).hexdigest()
    job_id = "rec_" + secrets.token_hex(16)
    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO recommendation_jobs
                (id,analysis_id,excluded_allergens,exclusions_hash,lang,force_fallback)
            VALUES (:id,:analysis_id,:allergens,:digest,:lang,:force_fallback)
            ON CONFLICT (analysis_id,exclusions_hash,lang) DO UPDATE SET
                status=CASE WHEN recommendation_jobs.status='failed' THEN 'queued' ELSE recommendation_jobs.status END,
                force_fallback=CASE WHEN recommendation_jobs.status='failed' THEN EXCLUDED.force_fallback ELSE recommendation_jobs.force_fallback END,
                last_error=CASE WHEN recommendation_jobs.status='failed' THEN NULL ELSE recommendation_jobs.last_error END,
                updated_at=now()
            RETURNING *
        """), {"id": job_id, "analysis_id": analysis_id, "allergens": PsycopgJson(allergens),
                 "digest": digest, "lang": lang, "force_fallback": force_fallback}).mappings().one()
        conn.execute(text("UPDATE analyses SET active_recommendation_job_id=:job,updated_at=now() WHERE id=:analysis"),
                     {"job": row["id"], "analysis": analysis_id})
    return dict(row)


def claim_recommendation_job():
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT r.*,a.result_json FROM recommendation_jobs r JOIN analyses a ON a.id=r.analysis_id
            WHERE r.status='queued' AND a.analysis_status='ready'
            ORDER BY r.created_at FOR UPDATE OF r SKIP LOCKED LIMIT 1
        """)).mappings().one_or_none()
        if row is None:
            return None
        conn.execute(text("""
            UPDATE recommendation_jobs SET status='processing',started_at=COALESCE(started_at,now()),
                heartbeat_at=now(),attempt_count=attempt_count+1,last_error=NULL,updated_at=now()
            WHERE id=:id AND status='queued'
        """), {"id": row["id"]})
        return dict(row)


def complete_recommendation_job(job_id, result):
    status = "fallback_ready" if result.get("fallback_used") else "ready"
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE recommendation_jobs SET status=:status,result_json=:result,strategy=:strategy,
                model=:model,fallback_used=:fallback,primary_error=:primary_error,
                completed_at=now(),heartbeat_at=now(),updated_at=now()
            WHERE id=:id AND status='processing'
        """), {"id": job_id, "status": status, "result": PsycopgJson(result),
                 "strategy": result.get("strategy"), "model": result.get("model"),
                 "fallback": bool(result.get("fallback_used")), "primary_error": result.get("primary_error")})


def fail_recommendation_job(job_id, error):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE recommendation_jobs SET status='failed',last_error=:error,
                completed_at=now(),heartbeat_at=now(),updated_at=now()
            WHERE id=:id AND status='processing'
        """), {"id": job_id, "error": str(error)[:2000]})


def block_recommendation_job(job_id):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE recommendation_jobs SET status='blocked',completed_at=now(),updated_at=now()
            WHERE id=:id AND status='queued'
        """), {"id": job_id})


def recover_stale_jobs(stale_after_seconds):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE analyses SET analysis_status='failed',analysis_stage='failed',
                analysis_error='Upload was interrupted before queueing.',updated_at=now()
            WHERE analysis_status='uploading'
              AND updated_at < now() - make_interval(secs => :seconds)
        """), {"seconds": stale_after_seconds})
        analyses = conn.execute(text("""
            UPDATE analyses SET analysis_status='queued',analysis_stage='queued',updated_at=now()
            WHERE analysis_status IN ('predicting_local','analyzing_regions')
              AND heartbeat_at < now() - make_interval(secs => :seconds)
              AND attempt_count < 2
        """), {"seconds": stale_after_seconds}).rowcount
        recs = conn.execute(text("""
            UPDATE recommendation_jobs SET status='queued',updated_at=now()
            WHERE status='processing' AND heartbeat_at < now() - make_interval(secs => :seconds)
              AND attempt_count < 2
        """), {"seconds": stale_after_seconds}).rowcount
        conn.execute(text("""
            UPDATE analyses SET analysis_status='failed',analysis_stage='failed',
                analysis_error='Analysis worker stopped after the retry limit.',updated_at=now()
            WHERE analysis_status IN ('predicting_local','analyzing_regions')
              AND heartbeat_at < now() - make_interval(secs => :seconds) AND attempt_count >= 2
        """), {"seconds": stale_after_seconds})
        conn.execute(text("""
            UPDATE recommendation_jobs SET status='failed',
                last_error='Recommendation worker stopped after the retry limit.',updated_at=now()
            WHERE status='processing'
              AND heartbeat_at < now() - make_interval(secs => :seconds) AND attempt_count >= 2
        """), {"seconds": stale_after_seconds})
    return analyses, recs


def get_recommendation_job(job_id, analysis_id=None):
    with engine.connect() as conn:
        sql, params = "SELECT * FROM recommendation_jobs WHERE id=:id", {"id": job_id}
        if analysis_id:
            sql += " AND analysis_id=:analysis_id"
            params["analysis_id"] = analysis_id
        row = conn.execute(text(sql), params).mappings().one_or_none()
    return dict(row) if row else None


def recommendation_response(row):
    result = dict(row.get("result_json") or {})
    return {"recommendation_job_id": row["id"], "recommendations": result.get("shades", []),
            "recommendations_reliable": result.get("reliable", False),
            "recommendations_catalog_source": result.get("catalog_source"),
            "recommendations_catalog_shades_considered": result.get("catalog_shades_considered"),
            "recommendations_status": row["status"],
            "recommendations_search_summary": result.get("search_summary", ""),
            "recommendations_model": row.get("model") or result.get("model"),
            "recommendations_strategy": row.get("strategy") or result.get("strategy"),
            "recommendations_fallback_used": bool(row.get("fallback_used")),
            "recommendations_primary_error": row.get("primary_error"),
            "recommendations_error": row.get("last_error")}
