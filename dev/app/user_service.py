import uuid

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.auth import hash_password
from app.db import engine


def ensure_users_table() -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    username VARCHAR(50) NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    first_name VARCHAR(100),
                    last_name VARCHAR(100),
                    age INTEGER,
                    skin_type_self_assessed VARCHAR(50),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT users_username_unique UNIQUE (username),
                    CONSTRAINT users_email_unique UNIQUE (email)
                )
                """
            )
        )


def create_user(req):
    user_id = str(uuid.uuid4())
    profile = req.profile
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO users (
                        id, username, email, password_hash, first_name, last_name,
                        age, skin_type_self_assessed
                    )
                    VALUES (
                        CAST(:id AS UUID), :username, :email, :password_hash,
                        :first_name, :last_name, :age, :skin_type
                    )
                    RETURNING id::text, username, email, first_name, last_name,
                              age, skin_type_self_assessed, created_at
                    """
                ),
                {
                    "id": user_id,
                    "username": req.username.strip(),
                    "email": req.email.lower().strip(),
                    "password_hash": hash_password(req.password),
                    "first_name": profile.first_name.strip() if profile and profile.first_name else None,
                    "last_name": profile.last_name.strip() if profile and profile.last_name else None,
                    "age": profile.age if profile else None,
                    "skin_type": profile.skin_type_self_assessed if profile else None,
                },
            ).mappings().one()
        return dict(row)
    except IntegrityError as exc:
        constraint = getattr(exc.orig, "diag", None)
        name = getattr(constraint, "constraint_name", "")
        if name == "users_email_unique":
            field = "email"
        elif name == "users_username_unique":
            field = "username"
        else:
            field = "email or username"
        raise ValueError(f"An account with this {field} already exists.") from exc


def find_user_by_email(email: str):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id::text, username, email, password_hash, first_name,
                       last_name, age, skin_type_self_assessed, created_at
                FROM users WHERE lower(email) = :email
                """
            ),
            {"email": email.lower().strip()},
        ).mappings().one_or_none()
    return dict(row) if row else None


def find_user_by_id(user_id: str):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id::text, username, email, first_name, last_name, age,
                       skin_type_self_assessed, created_at
                FROM users WHERE id = CAST(:id AS UUID)
                """
            ),
            {"id": user_id},
        ).mappings().one_or_none()
    return dict(row) if row else None
