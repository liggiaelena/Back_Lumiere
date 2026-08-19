import logging
import secrets
from psycopg2.extras import Json as PsycopgJson
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import engine

logger = logging.getLogger(__name__)


def ensure_analysis_ownership() -> None:
    """Add account ownership to databases created before history was introduced."""
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE analyses ADD COLUMN IF NOT EXISTS user_id UUID"))
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS analyses_user_created_idx "
                "ON analyses (user_id, created_at DESC)"
            )
        )


def save_analysis(result: dict, user_id: str | None = None) -> str:
    analysis_id = "ana_" + secrets.token_hex(4)
    with Session(engine) as session:
        session.execute(
            text("""
                INSERT INTO analyses
                    (id, user_id, tom_geral_fitzpatrick, tom_geral_hex, fitzpatrick_source, subtom_predominante, result_json)
                VALUES
                    (:id, CAST(:user_id AS UUID), :fitzpatrick, :hex, :source, :subtom, :result_json)
            """),
            {
                "id":          analysis_id,
                "user_id":     user_id,
                "fitzpatrick": result.get("tom_geral_fitzpatrick"),
                "hex":         result.get("tom_geral_hex"),
                "source":      result.get("fitzpatrick_source"),
                "subtom":      result.get("subtom_predominante"),
                "result_json": PsycopgJson(result),
            },
        )
        session.commit()
    logger.info("Saved analysis to database (id=%s)", analysis_id)
    return analysis_id


def list_user_analyses(user_id: str) -> list[dict]:
    with Session(engine) as session:
        rows = session.execute(
            text(
                """
                SELECT id, created_at, tom_geral_fitzpatrick, tom_geral_hex,
                       subtom_predominante
                FROM analyses
                WHERE user_id = CAST(:user_id AS UUID)
                ORDER BY created_at DESC
                """
            ),
            {"user_id": user_id},
        ).mappings().all()
    return [
        {**dict(row), "created_at": row.created_at.isoformat()}
        for row in rows
    ]


def get_analysis(analysis_id: str, user_id: str | None = None) -> dict | None:
    with Session(engine) as session:
        row = session.execute(
            text(
                """
                SELECT id, created_at, result_json FROM analyses
                WHERE id = :id
                  AND (user_id IS NULL OR user_id = CAST(:user_id AS UUID))
                """
            ),
            {"id": analysis_id, "user_id": user_id},
        ).fetchone()

    if row is None:
        logger.info("Analysis not found in database (id=%s)", analysis_id)
        return None

    data: dict = dict(row.result_json)
    data["id"] = row.id
    data["created_at"] = row.created_at.isoformat()
    logger.info("Loaded analysis from database (id=%s)", analysis_id)
    return data
