import secrets
from psycopg2.extras import Json as PsycopgJson
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import engine


def save_analysis(result: dict) -> str:
    analysis_id = "ana_" + secrets.token_hex(4)
    with Session(engine) as session:
        session.execute(
            text("""
                INSERT INTO analyses
                    (id, tom_geral_fitzpatrick, tom_geral_hex, fitzpatrick_source, subtom_predominante, result_json)
                VALUES
                    (:id, :fitzpatrick, :hex, :source, :subtom, :result_json)
            """),
            {
                "id":          analysis_id,
                "fitzpatrick": result.get("tom_geral_fitzpatrick"),
                "hex":         result.get("tom_geral_hex"),
                "source":      result.get("fitzpatrick_source"),
                "subtom":      result.get("subtom_predominante"),
                "result_json": PsycopgJson(result),
            },
        )
        session.commit()
    return analysis_id


def get_analysis(analysis_id: str) -> dict | None:
    with Session(engine) as session:
        row = session.execute(
            text("SELECT id, created_at, result_json FROM analyses WHERE id = :id"),
            {"id": analysis_id},
        ).fetchone()

    if row is None:
        return None

    data: dict = dict(row.result_json)
    data["id"] = row.id
    data["created_at"] = row.created_at.isoformat()
    return data
